# vLLM + LMCache + hipFile Benchmark

Benchmark kit for evaluating **KV-cache offload strategies**
in [vLLM](https://github.com/vllm-project/vllm) on AMD
MI325X GPUs. Three serving modes are compared:

| Mode | Script | KV cache location |
|------|--------|-------------------|
| Baseline | `serve_nocache.sh` | GPU VRAM only |
| CPU | `serve_cpu_cache.sh` | Host RAM via LMCache |
| AIS | `serve_ais_cache.sh` | NVMe via hipFile/GDS |

GPU memory utilisation is deliberately capped (30-32 %) to
create VRAM pressure and force KV-cache spill.

## Prerequisites

* Docker with GPU pass-through (ROCm `/dev/kfd`,
  `/dev/dri`)
* `/dev/infiniband` access (for RDMA / AIS mode)
* `/data` mount for NVMe-backed AIS storage
* A Hugging Face token with access to
  `openai/gpt-oss-120b`

## Quick start

### 1. Fetch the benchmark corpus

The text corpus is not stored in git. Download it with:

```bash
./scripts/fetch-corpus.sh
```

This pulls two novels from Project Gutenberg and writes
`scripts/configs/books.txt` (~6 MB).

### 2. Build the Docker image

```bash
docker buildx build -f Dockerfile \
    -t $(whoami)-hipfile .
```

### 3. Launch the container

```bash
docker run -it --rm \
    --device /dev/kfd \
    --device /dev/dri \
    --device /dev/infiniband \
    --security-opt apparmor=unconfined \
    --security-opt seccomp=unconfined \
    --network host --ipc host \
    --shm-size=10G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --cap-add IPC_LOCK \
    --cap-add SYS_PTRACE \
    --cap-add SYS_ADMIN \
    --cap-add BPF \
    --cap-add PERFMON \
    -v "$HOME:$HOME" \
    -v /data:/data \
    --env-file ~/docker.env \
    -v /lib/modules:/lib/modules:ro \
    -v /usr/src:/usr/src:ro \
    -v /etc/localtime:/etc/localtime:ro \
    -v /sys/kernel/debug/:/sys/kernel/debug/ \
    -v /sys/kernel/btf:/sys/kernel/btf:ro \
    "$(whoami)-hipfile"
```

If your Docker/kernel setup does not support the fine-grained
`BPF`/`PERFMON` capabilities and you still encounter permission
errors when collecting BPF traces, you can temporarily add
`--privileged` to the `docker run` command as a last resort.

### 4. Serve the model

Pick one of the three modes from inside the container:

```bash
# --gpu-memory-utilization and --tensor-parallel-size vLLM parameters can be set via env variables, such as: 
export GPU_MEMORY_UTILIZATION=0.5
export TENSOR_PARALLEL_SIZE=$(rocm-smi -i | grep "Instinct" | wc -l)

# Baseline (no external cache)
./scripts/serve_nocache.sh

# LMCache -> host CPU RAM
./scripts/serve_cpu_cache.sh

# LMCache -> NVMe via hipFile (AIS)
./scripts/serve_ais_cache.sh
```

### 5. Run benchmarks

In a second shell inside the same container:

```bash
# Short multi-turn (24 conversations, 12-18 turns)
./scripts/bench_multi_turn_short.sh

# Long multi-turn (240 conversations, 24-36 turns)
./scripts/bench_multi_turn_long.sh
```

### 6. Collect hipFile traces with BPF

```bash
# Trace hipFile IO with BPF
./scripts/trace_hipfile.sh [<custom libhipfile.so location>] > data.csv
```

## Directory layout

```
Dockerfile
scripts/
  fetch-corpus.sh              download corpus
  serve_nocache.sh             baseline serving
  serve_cpu_cache.sh           LMCache -> CPU RAM
  serve_ais_cache.sh           LMCache -> hipFile/AIS
  bench_multi_turn_short.sh    short benchmark
  bench_multi_turn_long.sh     long benchmark
  trace_hipfile.sh             BPF tracing script
  hipfile.bt                   BPF tracing recipe
  configs/
    lmcache-cpu.yaml           CPU cache config
    lmcache-ais.yaml           AIS cache config
    generate_multi_turn_short.json
    generate_multi_turn_long.json
    books.txt                  (generated, gitignored)
```

## References

* [hipFile](https://github.com/glimchb/hipFile) -- AMD
  GPU Direct Storage library
* [LMCache](https://github.com/glimchb/LMCache)
  (hipFile branch) -- KV-cache management for LLM
  inference
* [vLLM](https://github.com/vllm-project/vllm) -- high
  throughput LLM serving engine
