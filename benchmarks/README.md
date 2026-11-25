# Benchmarks

Every benchmark client runs from one image, `aic-client`
(`docker/client/Dockerfile`: Python only, no ROCm), through one command,
`aic-bench`. The stack starts a long-lived `<stack>-client` container on its
bridge with the repo at `/repo` and the logs at `/logs`.

```bash
make client-build                          # build the image (once)
make bench-list                            # the subcommands
make bench BENCH=agent-simple BENCH_ARGS="s1 220 16 480"
source scripts/aic-endpoints.sh            # sets AIC_STACK, this checkout's stack
docker exec "${AIC_STACK}-client" aic-bench cliff --arm kvd_v2 --concurrencies 1,2,4
docker exec "${AIC_STACK}-client" aic-bench cliff --help   # the underlying tool's help
```

## aic-bench

```text
aic-bench [--endpoint URL] [--model ID] [--out DIR] <subcommand> [args...]
```

Everything after the subcommand goes to the underlying tool unchanged.
`aic-bench` only fills in the endpoint, the model and an output location, in
the tool's own flag or environment variable, and never when you passed that
flag (or set that variable) yourself. The one exception is `reset-test`, where
`--endpoint` and `--model` on the command line take precedence over
`AIC_TEST_VLLM_URL` and `AIC_TEST_MODEL` in the environment.

| Option | Variable | Default |
| --- | --- | --- |
| `--endpoint` | `AIC_BENCH_ENDPOINT` | `http://aic-lmcache:8000` (server base URL, no `/v1`) |
| `--model` | `AIC_BENCH_MODEL` | the first model `GET <endpoint>/v1/models` lists |
| `--out` | `AIC_BENCH_OUT` | `/logs/bench`; a run writes to `<out>/<subcommand>/<UTC timestamp>/` |

The endpoint is a name on the `aic` bridge, never `localhost`: vLLM shares
`aic-lmcache`'s network namespace, so it answers on that name. `make bench`
resolves the endpoint of `<stack>-vllm-gpu$(GPU)` for you. Services next to the
endpoint are taken from its host unless a variable names them:

| Variable | Default | Used by |
| --- | --- | --- |
| `AIC_BENCH_LMCACHE_HTTP_URL` | `http://<endpoint host>:8080` | `reset-test`, `agent-kv nvme-sweep` |
| `AIC_BENCH_LMCACHE_METRICS_URL` | `http://<endpoint host>:9091` | `agent-simple`, `replay-hits` |
| `AIC_BENCH_NIXL_METRICS_URL` | `http://<endpoint host>:19090` | `agent-simple` |
| `AIC_HSA_SNOOP_URL` | `http://aic-hsa-snoop:9488` | `fixed-load` |

Scripts are run from `$AIC_BENCH_ROOT` if set, else from `/repo/benchmarks`
when it holds `aic_bench.py` (so an edit in the checkout needs no image
rebuild), else from the copy baked into the image at `/opt/aic/benchmarks`.

## Subcommands

| Subcommand | Runs | What `aic-bench` fills in |
| --- | --- | --- |
| `list` | | prints this table's first column with one-line descriptions |
| `env` | | prints the resolved endpoint, model (asking the endpoint when none is set), output root and benchmark root |
| `cliff` | `run_cliff.py`: concurrency sweep, shared or per-client prefix | `--endpoint`, `--model`, `--out <run dir>/<arm>.csv` |
| `plot` | `plot_cliff.py`: PNG charts from cliff CSVs | `--output-dir <run dir>` |
| `cliff-report` | `cliff_to_html.py`: HTML page from the run directories of the cliff Slurm jobs (`results/cliff-*.csv` and `plots/`), not from an `aic-bench cliff` run | `--output-dir <run dir>` |
| `reset-test` | `vllm_reset_test.py`: L1 and L2 retrieval after `/reset_prefix_cache` | `AIC_TEST_VLLM_URL`, `AIC_TEST_LMCACHE_URL`, `AIC_TEST_MODEL` |
| `agent-simple` | `agent-simple/stress.py`: synthetic mixed-size cache-tier stress | `AIC_BENCH_ENDPOINT` |
| `replay-hits` | `common/replay_hits.py`: evict prompts from the GPU cache, replay for L1/L2 hits | `AIC_BENCH_ENDPOINT` |
| `fixed-load` | `common/fixed_load.py`: fixed workload for comparing hsa-snoop builds | `AIC_BENCH_ENDPOINT` |
| `serve-sim` | `aic_bench_serve.py`: multi-turn ShareGPT sessions with think time | `--base-url`, `--model`, `--output <run dir>` |
| `agent-kv` | the `agent-kv` console script: `run`, `run-ramp`, `nvme-sweep` | `--base-url <endpoint>/v1`, `--model`, `--output <run dir>/<name>.json`, `--cache-dir <out>/agent-kv/books`; `--lmcache-http-url` for `nvme-sweep` |
| `aiperf` | the `aiperf` console script | nothing |

`AIC_BENCH_ENDPOINT`, `AIC_BENCH_OUT` and, when known, `AIC_BENCH_MODEL` are
exported to every tool. `agent-simple`, `replay-hits` and `fixed-load` share
`common/kv_client.py`, which reads them. `agent-kv` gets no `--base-url` when
you pass `--host` or `--port`.

## NVMe delay sweep

`agent-kv nvme-sweep` measures the TTFT of L2 hits against injected NVMe read
latency and fill fraction. The latency comes from a `dm-delay` device under the
L2 mount, which agent-kv adjusts with `sudo dmsetup`. The client container is
unprivileged, so `agent-kv/nvme_delay_sweep.sh` runs the sweep with
`agent-kv/shim/sudo` on the client's `PATH`: the shim hands each change to the
script, which applies it on the host with `sudo` and then drops the L2 files
from the page cache, because the stack's L2 writes are buffered and a cached
read never reaches the device.

```bash
# Once, with the stack down; `down` puts the plain device back.
benchmarks/agent-kv/dm_delay_setup.sh up /mnt/<L2 mount point>

# The sweep resets the GPU prefix cache, which drops stores that lazy offload
# has deferred, so start the stack without it.
make up LMCACHE_LAZY_OFFLOAD=false NVME_DATA=<L2 dir>

make agent-kv-nvme-sweep NVME_DATA=<L2 dir> \
  AGENT_KV_SWEEP_ARGS="--delays 0,10,50 --fill-fractions 0.5,1.0 --num-books 4"
```

Without the script, `aic-bench agent-kv nvme-sweep` still runs, with no latency
injected.

```bash
aic-bench reset-test
aic-bench replay-hits r1 90 440 12 32 3 --reset-gpu
aic-bench agent-kv run --num-books 2 --rounds 3
aic-bench agent-kv nvme-sweep --num-books 2
aic-bench --endpoint http://aic-kvbench:<port> cliff --arm kvbench
aic-bench plot --input /logs/bench/cliff/<UTC timestamp>
```

## AgentX

The AgentX benchmark (`aiperf profile --scenario inferencex-agentx-mvp`) has
its own host-side runner, because each run gets a fresh container with the
corpus and the run directory mounted: see [`agentx/README.md`](agentx/README.md)
and `make agentx-run`.

## Not in this image

The accuracy gate (`tests/accuracy`, `lm_eval`) pulls in torch and stays out of
`aic-client`.
