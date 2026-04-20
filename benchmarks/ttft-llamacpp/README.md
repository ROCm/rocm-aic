# llama.cpp TTFT Benchmark

Measure **Time-To-First-Token (TTFT)** on AMD ROCm GPUs using
[llama.cpp][llamacpp]'s built-in slot save/restore API.  Compares
cold prefill (no cache) against warm restore (slot loaded from
tmpfs/RAM or disk) across multiple context sizes.

This benchmark works on both AMD Instinct (CDNA) and Radeon
(RDNA) GPUs, since llama.cpp supports both via HIP.

## Quick start (Slurm)

```bash
sbatch .slurm/ttft-llamacpp.sbatch
```

The job auto-detects the GPU, downloads the model if needed,
builds llama.cpp (cached for subsequent runs), and sweeps
three context sizes with 10 repeats each.  Results and
graphs land in `.slurm/logs/`.

Override defaults via env vars:

```bash
REPEATS=5 SEED=123 sbatch .slurm/ttft-llamacpp.sbatch
```

## Manual usage

```bash
# install deps
pip install openai matplotlib

# run the benchmark
python3 ttft_bench.py run \
    --model /path/to/model.gguf \
    --context-chars 400 4000 40000 \
    --repeats 10 \
    --output results.jsonl

# generate report from existing results
python3 ttft_bench.py report results.jsonl \
    --gpu "AMD Instinct MI308X"
```

## How it works

For each context size:

1. **Cold runs** -- start a fresh llama-server, send a long
   prompt, measure TTFT from scratch, save the slot to disk.
   Server is restarted between each repeat.
2. **Warm tmpfs** -- copy the slot file to `/dev/shm` (RAM),
   start the server once, then repeat: erase slot, restore
   from tmpfs, measure TTFT.
3. **Warm disk** -- same but restore from a local disk path
   with page cache evicted between repeats.

Warm phases keep the server running across repeats (slot
erase + restore clears KV state without restart), cutting
total runtime significantly.

## Output

The `run` subcommand produces:

- `results.jsonl` -- per-measurement records with TTFT,
  disk IO, startup timing, GPU, and model metadata
- Summary table printed to stdout
- `summary.csv`, `ttft_bars.png`, `speedup.png`,
  `disk_io.png`, `report_meta.json`

Example output:

```
GPU:   AMD Instinct MI308X
Model: Qwen3-8B-Q4_K_M.gguf

ctx_chars  phase            n    mean_ms     min_ms     max_ms  speedup
-----------------------------------------------------------------------
400        cold            10      156.5      151.6      161.7
400        warm-tmpfs      10       90.7       86.2       94.3     1.7x
400        warm-disk       10       88.7       85.9       92.3     1.8x
4000       cold            10      779.3      643.4      946.4
4000       warm-tmpfs      10       93.8       90.9       95.7     8.3x
4000       warm-disk       10       93.8       88.6       96.3     8.3x
40000      cold            10    12724.6    12409.4    14722.9
40000      warm-tmpfs      10      123.5      114.1      132.6   103.0x
40000      warm-disk       10      120.9      110.7      131.7   105.3x
```

## Build-time optimisation

The sbatch caches the llama.cpp build at
`~/.cache/llama.cpp-{tag}-{arch}`.  First run builds with
`-DGGML_HIP_FA=OFF` (disables flash attention templates,
~2 min build).  Subsequent runs skip the build entirely.

Set `LLAMACPP_FA=ON` to enable flash attention (slower build,
slightly better decode throughput -- does not affect TTFT).

## cache-disk patch

The `patches/0001-cache-disk.patch` adds `--cache-disk` and
`--cache-disk-max` to llama-server for automatic disk-tier
prompt caching.  See the patch header for details.

## Directory layout

```
Dockerfile                    ROCm + llama.cpp (HIP) build
README.md                     This file
requirements.txt              Python deps (openai, matplotlib)
ttft_bench.py                 Unified CLI: run + report
corpus.txt                    (generated, gitignored)
scripts/
  docker-build.sh             Build the Docker image
  docker-run.sh               Launch with ROCm flags
  fetch-corpus.sh             Download Gutenberg corpus
patches/
  0001-cache-disk.patch       --cache-disk feature patch
```

<!-- References -->

[llamacpp]: https://github.com/ggml-org/llama.cpp
