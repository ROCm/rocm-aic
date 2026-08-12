# Key Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `ROCM_ARCH` | auto-detected | GPU arch, e.g. `gfx942` |
| `HF_TOKEN` | — | HuggingFace access token (required) |
| `VLLM_MODEL` | `openai/gpt-oss-120b` | Model to serve |
| `NVME_DATA` | `/mnt/lmcache-nvme` | Host path for NVMe L2a pool |
| `NFS_DATA` | `/mnt/lmcache-nfs` | Host path for NFS-over-RDMA L2b pool |
| `GDS_SLAB_DATA` | — | Host path for GDS NVMe slab (GDS L1 mode only) |
| `GDS_MODE` | — | Set to `1` to enable GDS L1 mode |
| `AIC_L2_BACKEND` | `nixl` | LMCache L2 backend: `nixl` (AIS_MT NVMe + POSIX NFS) or `local_disk` (native LocalDiskBackend via a mounted config) |
| `KV_TRANSFER_ARG` | LMCacheMPConnector JSON | vLLM `--kv-transfer-config` arg; empty = plain vLLM (the cliff `vram` baseline). Wrap the JSON in single quotes |
| `VLLM_IPC_MODE` | `service:lmcache` | vLLM IPC namespace mode. Cache-backed runs join LMCache's shareable namespace; use `shareable` when launching vLLM alone |
| `AIC_SHM_SIZE` | `64gb` | Capacity of the private `/dev/shm` used by LMCache and vLLM; increase if LMCache's L1 request is larger |
| `LMCACHE_L1_SIZE_GB` | `20` | MP server L1 cap in GiB (DRAM L1 in nvme mode, hipFile slab size in GDS mode) |
| `LMCACHE_NVME_POOL` | `4096` | NIXL pool slots for NVMe adapter |
| `LMCACHE_NVME_SLOT_SIZE` | `268435456` | NIXL file size per NVMe pool slot, bytes (256 MiB) |
| `LMCACHE_NFS_POOL` | `1024` | NIXL pool slots for NFS adapter |
| `VLM_BLOCK_SIZE` | `64` | vLLM `--block-size`; set this to match the model and connector cache geometry |
| `VLM_ATTENTION_BACKEND` | `TRITON_ATTN` | vLLM `--attention-backend` (TRITON_ATTN supports KV connectors) |
| `VLM_KV_CACHE_DTYPE` | `fp8` | vLLM `--kv-cache-dtype` (`auto` for non-fp8 arches) |
| `VLLM_EXTRA_ARGS` | — | Extra vLLM args appended verbatim (e.g. `--hf-overrides '{...}'`; single-quote embedded JSON) |
| `COMPOSE_PLUGIN_VERSION` | `v2.40.0` | docker compose v2 plugin version installed by `make ensure-compose` / `ensure_compose` when missing |
| `AIC_TINY_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Model served by `make tiny-test` (end-to-end serve check) |
| `HF_HOME` | `~/.cache/huggingface` (`/shared_nfs/huggingface` on SPUR) | Persistent Hugging Face cache used by normal, cliff, monitoring, and tiny-model paths |
| `TENSOR_PARALLEL_SIZE` | `1` | vLLM tensor parallel degree |
| `GPU` | `0` | Local fallback GPU when no scheduler visibility variable is set. |
| `AIC_ROCR_VISIBLE` | resolved GPU set | `ROCR_VISIBLE_DEVICES` for every container with `/dev/kfd`. **Absolute** host GPU IDs (e.g. `3` or `2,5`). On standard Slurm, preserves `ROCR_VISIBLE_DEVICES`, then `HIP_VISIBLE_DEVICES`, then `CUDA_VISIBLE_DEVICES`; on SPUR it comes from the controller allocation and an unresolvable allocation fails the job. |
| `AIC_HIP_VISIBLE` | relative to `AIC_ROCR_VISIBLE` | `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`. **Relative** indices `0..n-1` (e.g. `0` or `0,1`) — HIP filters the set ROCR already filtered. Set together with `AIC_ROCR_VISIBLE`. |
| `AIC_CLIFF_GPUS` | `1` | GPUs reserved by a SPUR `make cliff-submit` (`--gpus=N`). Raise only alongside a matching `TENSOR_PARALLEL_SIZE` |
| `AIC_TEST_GPUS` | `1` | GPUs reserved by SPUR `smoke-test` and `tiny-test` jobs (`--gpus=N`). Standard Slurm continues to use `AIC_TEST_GRES`. |
| `AIC_NVME_AUTO` | `1` (cliff) | Auto-detect a dedicated local NVMe for the LMCache tiers: reuse a mounted `aic-lmcache` volume, else format+mount a raw non-root spare, else use a non-root mounted NVMe, else node-local `/tmp`. `0` forces `/tmp`; needs passwordless `sudo` to format/mount |
| `AIC_NVME_MOUNT` | `/mnt/aic-lmcache` | Mountpoint used when auto-provisioning a spare NVMe (left mounted for reuse) |
| `AIC_MONITORING` | `1` | Auto-start the Prometheus sidecar in cliff sbatch runs (`0` to skip) |
| `AIC_METRICS_DIR` | `logs/<job-id>/prometheus` (cliff) | Prometheus TSDB dir — bind-mount an NFS path here |
| `AIC_EXPORTERS` | `1` (cliff) / `0` (make) | Also launch containerized node + AMD GPU exporters |

## Accuracy test (`make accuracy-test`)

The KV-integrity gate: scores gsm8k against a VRAM-only arm and a tiered arm in
one job and asserts that routing KV through DRAM/NVMe did not change the
answers. See [`tests/accuracy/README.md`](../tests/accuracy/README.md) for what
each assertion catches and how to add a model.

| Variable | Default | Description |
| --- | --- | --- |
| `AIC_ACCURACY_MODEL` | `$AIC_TINY_MODEL` | Model to score. Must match what the arms serve |
| `AIC_ACCURACY_LIMIT` | `0` (full split) | Cap on gsm8k items. Lowers wall clock and resolution together — the score's standard error grows as `1/sqrt(LIMIT)` |
| `AIC_ACCURACY_DELTA` | `0.02` | How far the tiered arm may fall below the baseline arm before it counts as corruption |
| `AIC_ACCURACY_SKIP_BASELINE` | `0` | `1` drops the VRAM-only arm (what `accuracy-test-fast` does): no differential, but the floor, liveness and restart assertions still run |
| `AIC_ACCURACY_RESTART_LIMIT` | `200` | Item cap for the post-restart re-score, which tests retrieval rather than statistical quality |
| `AIC_ACCURACY_FAST_LIMIT` | `200` | `AIC_ACCURACY_LIMIT` used by `make accuracy-test-fast` |
| `AIC_ACCURACY_CONCURRENT` | `32` | `lm_eval` request concurrency |
| `AIC_ACCURACY_BASELINE_URL` | — | Baseline arm base URL, e.g. `http://172.18.0.4:8000/v1`. Only needed when running pytest by hand; the driver supplies it |
| `AIC_ACCURACY_TIERED_URL` | — | Tiered arm base URL. Unset or unreachable endpoints skip cleanly rather than failing |
| `AIC_ACCURACY_BASELINE_SCORE` | — | Pre-measured baseline score. The arms share a container name, a port and the GPU, so the driver scores them sequentially and passes the first arm's number forward |
| `AIC_ACCURACY_REFERENCE_SCORE` | — | Pre-restart tiered score; enables the restart assertion |
| `AIC_ACCURACY_SCORE_OUT` | — | Write the measured tiered score here as JSON |
| `AIC_ACCURACY_TIME` | `01:30:00` | Slurm wall-time for the accuracy job |
| `AIC_ACCURACY_CPUS` | `8` | Slurm `--cpus-per-task` |
| `AIC_ACCURACY_MEM` | `32G` | Slurm `--mem` |
| `AIC_ACCURACY_READY_TIMEOUT` | `120` | Endpoint readiness attempts, 5s apart (so up to 10 min per arm) |
