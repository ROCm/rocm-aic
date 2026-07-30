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
| `LMCACHE_L1_SIZE_GB` | `20` | MP server L1 cap in GiB (DRAM L1 in nvme mode, hipFile slab size in GDS mode) |
| `LMCACHE_NVME_POOL` | `4096` | NIXL pool slots for NVMe adapter |
| `LMCACHE_NVME_SLOT_SIZE` | `268435456` | NIXL file size per NVMe pool slot, bytes (256 MiB) |
| `LMCACHE_NFS_POOL` | `1024` | NIXL pool slots for NFS adapter |
| `VLM_ATTENTION_BACKEND` | `TRITON_ATTN` | vLLM `--attention-backend` (TRITON_ATTN supports KV connectors) |
| `VLM_KV_CACHE_DTYPE` | `fp8` | vLLM `--kv-cache-dtype` (`auto` for non-fp8 arches) |
| `VLLM_EXTRA_ARGS` | — | Extra vLLM args appended verbatim (e.g. `--hf-overrides '{...}'`; single-quote embedded JSON) |
| `COMPOSE_PLUGIN_VERSION` | `v2.40.0` | docker compose v2 plugin version installed by `make ensure-compose` / `ensure_compose` when missing |
| `AIC_TINY_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Model served by `make tiny-test` (end-to-end serve check) |
| `AIC_TINY_HF_HOME` | `<image-dir>/tiny-hf` | Persistent HF cache the tiny model is downloaded into for `tiny-test` |
| `TENSOR_PARALLEL_SIZE` | `1` | vLLM tensor parallel degree |
| `GPU` | `0` | ROCR_VISIBLE_DEVICES for the vllm container |
| `AIC_NVME_AUTO` | `1` (cliff) | Auto-detect a dedicated local NVMe for the LMCache tiers: reuse a mounted `aic-lmcache` volume, else format+mount a raw non-root spare, else use a non-root mounted NVMe, else node-local `/tmp`. `0` forces `/tmp`; needs passwordless `sudo` to format/mount |
| `AIC_NVME_MOUNT` | `/mnt/aic-lmcache` | Mountpoint used when auto-provisioning a spare NVMe (left mounted for reuse) |
| `AIC_MONITORING` | `1` | Auto-start the Prometheus sidecar in cliff sbatch runs (`0` to skip) |
| `AIC_METRICS_DIR` | `logs/<job-id>/prometheus` (cliff) | Prometheus TSDB dir — bind-mount an NFS path here |
| `AIC_EXPORTERS` | `1` (cliff) / `0` (make) | Also launch containerized node + AMD GPU exporters |

## Emulation mode and profile capture

See [EMULATE.md](EMULATE.md) for the workflow these belong to.

| Variable | Default | Description |
| --- | --- | --- |
| `VLLM_EMULATOR_ENABLE_ORACLE` | — | `1` turns on emulation: platform plugin, `torch.cuda` shims, model-load stubs. Unset = ordinary serve |
| `VLLM_EMULATOR_EXECUTOR_HOOK` | — | `1` swaps `UniProcExecutor.execute_model()` for the profile-pack latency draw |
| `VLLM_EMULATOR_PROFILE_PACK` | `/opt/llm-emu/profiles/MI300X-Qwen3-8B.json` | Profile pack to replay (in-image path, or `/profiles/<pack>.json` with `EMU_PROFILE_PACK_HOST`) |
| `VLLM_EMULATOR_MODE` | `realtime` | `realtime` sleeps for the predicted latency; `accelerated` resolves immediately |
| `VLLM_EMULATOR_ORACLE_K` | `1` | Oracle neighbor count (`auto` for adaptive) |
| `VLLM_EMULATOR_MEMORY` | from pack | Override the emulated device memory, bytes. Pass only when set — an empty value is not the same as unset |
| `VLLM_EMULATOR_DEBUG` | — | `1` logs every oracle lookup |
| `VLLM_EMULATOR_TRACE_STEP_CYCLE` | — | `1` records one step-latency sample per step on **real** hardware (profile capture); does not enable emulation |
| `VLLM_EMULATOR_STEP_TRACE_OUTPUT` | `/tmp/emulator_step_trace.jsonl` | Where that trace is written |
| `EMU_PROFILE_PACK_HOST` | `/tmp` | Host dir bind-mounted at `/profiles` in the emulator container |
| `AIC_EMULATE_IMAGE` | `rocm-aic:7.14-emulate` | Image tag built by `make dist-build-emulate` |
| `AIC_EMULATE_ARCH` | `gfx942` | `ROCM_ARCH` for the emulation build (only selects torch's wheel extra) |
| `AIC_EMULATE_VLLM_DEVICE` | `empty` | `VLLM_TARGET_DEVICE` for that build: `empty` compiles no GPU kernels, `rocm` builds them |
| `AIC_EMULATE_MODEL` | `Qwen/Qwen3-8B` | Model served by `make emulate-test` |
| `AIC_EMULATE_PROFILE_PACK` | `/opt/llm-emu/profiles/MI300X-Qwen3-8B.json` | Pack used by `make emulate-test` |
| `AIC_EMULATE_PACK_HOST` | — | Host dir of packs to mount for `make emulate-test` / `emulate-validate` |
| `AIC_EMULATE_TEST_NODE` / `AIC_EMULATE_TEST_CONSTRAINT` | — / `CPUONLY` | Node selection for the CPU-only emulation jobs |
| `AIC_CAPTURE_MODEL` | `Qwen/Qwen3-8B` | Model served during `make profile-capture` |
| `AIC_CAPTURE_SWEEP` | 9 points | `input_len,output_len,concurrency,num_prompts` list driving the capture |
| `AIC_CAPTURE_DIR` | `<image-dir>/profiles` | Where pack, trace, provenance and benchmark results land |
| `AIC_CAPTURE_HF_HOME` | `<image-dir>/capture-hf` | HF cache for the capture serve (real weights, so size it accordingly) |
| `AIC_CAPTURE_CONSTRAINT` / `AIC_CAPTURE_NODE` | `GFX942` / — | GPU node selection for the capture job |
| `AIC_CAPTURE_MAX_MODEL_LEN` / `AIC_CAPTURE_MAX_BATCHED_TOKENS` / `AIC_CAPTURE_GPU_UTIL` | `8192` / `4096` / `0.85` | Serve flags baked into the captured pack |
| `AIC_VALIDATE_PACK` | — | Pack scored by `make emulate-validate` (required) |
| `AIC_VALIDATE_SWEEP` | 3 points | Benchmark points replayed for the real-vs-emulated diff |
| `AIC_VALIDATE_REAL_DIR` | `<pack dir>/bench` | Real-hardware benchmark results to diff against |
