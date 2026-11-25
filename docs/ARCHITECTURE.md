# Architecture

How the AIC containers fit together at runtime. For how to start them, see
[QUICK_START.md](QUICK_START.md); for every variable, see
[ENVIRONMENT.md](ENVIRONMENT.md).

## Images

`docker/docker-bake.hcl` builds every image and owns every version pin.

| Image | Contents |
| --- | --- |
| `aic-base` | ROCm base image plus the source-built PyTorch, Triton and NIXL shared by the two below |
| `aic-vllm` | `aic-base` plus vLLM, AITER and FlashAttention |
| `aic-lmcache` | `aic-base` plus LMCache |
| `aic-client` | The benchmark clients behind `aic-bench`; Python only, no ROCm |

## Containers

`docker/docker-compose.yml` is the single launch path for `make`, the Slurm
jobs and CI.

| Container | Role |
| --- | --- |
| `aic-lmcache` | LMCache multi-process server: the DRAM L1 and the NVMe/NFS L2 tiers |
| `aic-vllm-gpu<N>` | vLLM, connected to LMCache over ZMQ |
| `aic-vllm-only-gpu<N>` | vLLM with no LMCache, for the cliff baseline arm |
| `aic-client` | Runs the benchmarks and test clients |
| `aic-prometheus`, `aic-grafana`, exporters | Monitoring; see [METRICS_TELEMETRY.md](METRICS_TELEMETRY.md) |

`vllm` joins `aic-lmcache`'s network, IPC and PID namespaces, because the GPU
IPC handles LMCache uses to read and write vLLM's KV cache travel over an
abstract socket that only works inside one network namespace. A
`vllm-only` container has namespaces of its own.

## Networking

Each stack has a name, `AIC_STACK` (`aic-<job id>` in a Slurm job, else a
`aic-<id>` generated once per checkout): it names
the compose project, a private Docker bridge of the same name, and prefixes
every container (`<stack>-lmcache`, ...). On its own bridge each container also
answers to its name in the table above, except `vllm`, which shares lmcache's
namespace and answers as `aic-lmcache` or `aic-vllm`. No container binds a
fixed host port, so several stacks can share a multi-tenant host:

- Only Grafana is published, on `127.0.0.1`. `make` picks the lowest free port
  from 3000; `make endpoints` prints the URL.
- Every client runs in a container on the bridge: `aic-client` for `make`
  targets, or a throwaway `docker run --network "${AIC_STACK}"` container.
  `scripts/aic-endpoints.sh` provides `aic_fetch` (one request from the bridge)
  and `aic_bridge_url` (the URL a containerized client should use). A process
  on the host cannot reach vLLM at `localhost:8000`.
- On the bridge, use the fixed aliases (`http://aic-lmcache:8000`); they
  resolve only within their own stack. Host-side `docker` commands need the
  prefixed container name.
