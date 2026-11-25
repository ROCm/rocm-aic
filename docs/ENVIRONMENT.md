# Environment Variables

This is the single reference for the variables that configure the AIC build,
the stack, and the Slurm jobs. The tables are generated from the files that
hold the defaults, so a default shown here is the one in code:

```bash
make env-docs         # regenerate the tables after changing a default
make env-docs-check   # what the lint workflow runs
```

`make list-vars` prints the values a `make` invocation would use on this host.

Descriptions and table membership live in
[`environment-vars.txt`](environment-vars.txt). The check fails until a variable
that code gives a default, or that compose interpolates, has an entry there; a
variable only read as plain `$(VAR)` / `${VAR}` in make or a script has to be
added by hand.

## Where a default lives

Each variable has one owner; the other layers pass on only what the user set.
Where an entry point keeps a default of its own (a preset for one job, a site
default), the tables show it next to the owner's.

| Layer | File | Owns |
| --- | --- | --- |
| Build | `docker/docker-bake.hcl` | Every build input: version pins, source URLs, base images, tool versions, image names and tags, cache settings. Other layers read the resolved values through `docker/scripts/aic-bake-env.sh` |
| Runtime | `docker/docker-compose.yml` | Container runtime defaults: what the services are started with |
| Make | `Makefile`, `mk/*.mk` | Variables a make recipe expands itself (host paths, ports it probes, benchmark and test knobs of the local targets) |
| Slurm | `.slurm/*`, `monitoring/monitoring-lib.sh` | Slurm resources (time, CPUs, memory, constraints) and the procedure of each job |
| AgentX | `benchmarks/agentx/*.sh` | The `AGENTX_*` knobs of the AgentX benchmark; the scripts also run without make |
| Client | `benchmarks/*.py` | The `AIC_BENCH_*` settings of `aic-bench`, each also a command-line option |
| Site | `.env.<cluster>` (not tracked in git) | Values that differ per cluster: partition, constraint, image and cache directories |

Precedence under `make`, highest first: the command line (`make VAR=value`),
`.env.<cluster>`, then the owner's default. Run directly, a Slurm script reads
the same file, and a variable exported in the environment beats it. The SPUR
block of the `Makefile` holds the site defaults every SPUR user needs; the
tables list them as `SPUR`.

Reading the tables:

- **Owner** is the file holding the default in the **Default** column.
- A second value in the **Default** column is a different default at another
  entry point, named in parentheses: a compose service, a Slurm script, or the
  `Makefile` on SPUR. `another branch` is a second default in the same script,
  chosen by a condition the description names.
- `(only when unset)` marks a default that an empty value does not bring
  back: setting the variable empty is a setting of its own (it disables the
  feature, or selects a private namespace).
- `(empty)` is an empty default, `(unset)` means nothing sets it, `(required)`
  means the script stops without it, and `(computed)` is derived in the owner.
- Bake defaults that refer to another variable are shown as written
  (`${ROCM_VERSION}`), as are make (`$(CURDIR)`) and shell (`${USER}`) ones.

## Deprecated aliases

Each is accepted for one release and then removed. Set the canonical name
only.

| Deprecated | Use | Accepted by |
| --- | --- | --- |
| `AIC_TLS_CERT` | `TLS_CERT` | `Makefile`, `.slurm/run-build-distribute.sh` |
| `AIC_NVME_EXPORTER_VERSION` | `NVME_EXPORTER_VERSION` | `.slurm/run-build-distribute.sh` |
| `AIC_RDMA_EXPORTER_VERSION` | `RDMA_EXPORTER_VERSION` | `.slurm/run-build-distribute.sh` |
| `AIC_ATTENTION_BACKEND` | `VLM_ATTENTION_BACKEND` | `.slurm/run-cliff.sbatch` |
| `AIC_KV_CACHE_DTYPE` | `VLM_KV_CACHE_DTYPE` | `.slurm/run-cliff.sbatch` |
| `KVBENCH_READY_TIMEOUT` | `KVBENCH_READY_S` | `.slurm/run-kvbench-cliff.sbatch` |

## Build

Any variable of `docker/docker-bake.hcl` can be set in the environment of
`make build`, `docker buildx bake`, or a Slurm build; the Dockerfiles hold no
defaults of their own, except `docker/kvbench/Dockerfile` and
`docker/client/Dockerfile` (compose also builds them), whose defaults a test
keeps equal to bake's.

### Image names and tags

<!-- BEGIN generated:image -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `IMAGE_TAG` | (computed); `latest` (`Makefile`) | `docker/docker-bake.hcl` | Version tag shared by the stack images, derived from `AIC_VERSION`, the pins and `ROCM_ARCH`. Set it to override the derivation; without `docker buildx` make falls back to `latest` |
| `AIC_VERSION` | `0.1.0` | `docker/docker-bake.hcl` | AIC release. The repo-root `VERSION` file owns it (make and `aic-bake-env.sh` export it from there); the bake literal is the fallback for a bare `docker buildx bake` |
| `AIC_IMAGE_PREFIX` | (empty) | `docker/docker-bake.hcl` | Registry prefix for every image tag, with its trailing `/` (e.g. `ghcr.io/rocm/rocm-aic/`). Empty keeps the bare local names |
| `AIC_IMAGE_EXTRA_TAG` | (empty) | `docker/docker-bake.hcl` | Short alias pushed next to the version tag (e.g. `gfx1250`); only used with `AIC_IMAGE_PREFIX` |
| `VLLM_IMAGE_REF` | `aic-vllm:latest` | `docker/docker-compose.yml` | Full `name:tag` of the vLLM image compose starts. `make` exports the version-tagged ref bake resolves; the default is the `:latest` alias a local bake adds |
| `LMCACHE_IMAGE_REF` | `aic-lmcache:latest` | `docker/docker-compose.yml` | Full `name:tag` of the LMCache image compose starts; resolved the same way as `VLLM_IMAGE_REF` |
| `HSA_SNOOP_IMAGE` | `aic-hsa-snoop:latest` | `docker/docker-compose.yml` | Full `name:tag` of the hsa-snoop image compose starts |
| `KVBENCH_IMAGE_REF` | `aic-kvbench:177cf4e` | `docker/docker-compose.yml` | Full `name:tag` of the KVBench image; bake tags it `aic-kvbench:<KVBENCH_IMAGE_TAG>` |
| `KVBENCH_IMAGE_TAG` | `substr(KVBENCH_REF, 0, 7)` | `docker/docker-bake.hcl` | Tag of the KVBench image, the short `KVBENCH_REF` |
| `AIC_VLLM_IMAGE` | `aic-vllm:${_aic_tag:-latest}` | `.slurm/run-build-distribute.sh` | Slurm scripts: `name:tag` of the vLLM image to build, store and run. The tag is the one bake derives |
| `AIC_LMCACHE_IMAGE` | `aic-lmcache:${_aic_tag:-latest}` | `.slurm/run-build-distribute.sh` | Slurm scripts: `name:tag` of the LMCache image |
| `AIC_HSA_SNOOP_IMAGE` | `aic-hsa-snoop:latest` | `.slurm/run-build-distribute.sh` | Slurm scripts: `name:tag` of the hsa-snoop image built by `build-exporters` and loaded by the cliff job |
| `AIC_BASE_IMAGE` | `aic-base:${IMAGE_TAG:-latest}` | `.slurm/run-build-distribute.sh` | Slurm scripts: `name:tag` the `bake` command stores the base image under |
| `AIC_IMAGE` | `${AIC_IMAGE_NAME:-rocm-aic}:${_aic_tag:-latest}`; `rocm-aic:latest` (`monitoring/monitoring-lib.sh`) | `.slurm/run-build-distribute.sh` | Slurm scripts: a complete `name:tag` for the single-image paths, used verbatim; wins over `AIC_IMAGE_NAME`. The monitoring library runs its Prometheus queries from it |
| `AIC_IMAGE_NAME` | `rocm-aic` | `.slurm/run-build-distribute.sh` | Slurm scripts: image name only; the version-derived tag is appended |
| `AIC_EMULATE_IMAGE` | `${IMAGE_NAME:-rocm-aic}:${IMAGE_TAG_EMULATE:-10.2-emulate}` | `.slurm/run-build-distribute.sh` | `name:tag` of the CPU-only emulation image built by `make dist-build-emulate` |
| `IMAGE_NAME` | `rocm-aic` | `.slurm/run-build-distribute.sh` | Name part of the default `AIC_EMULATE_IMAGE` |
| `IMAGE_TAG_EMULATE` | `10.2-emulate` | `.slurm/run-build-distribute.sh` | Tag part of the default `AIC_EMULATE_IMAGE` |
<!-- END generated:image -->

### ROCm and GPU target

<!-- BEGIN generated:rocm -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `ROCM_VERSION` | `10.2.0`; `unknown` (`docker/docker-compose.yml`) | `docker/docker-bake.hcl` | ROCm release the images are built on |
| `ROCM_ARCH` | (empty); `gfx950` (`Makefile`, SPUR); `$(detect_gpu_arch)` (`.slurm/run-cliff.sbatch`) | `docker/docker-bake.hcl` | GPU arch of this host's stack: a `;`-, `,`- or space-separated `gfx` list. `make` fills it from `rocm_agent_enumerator` when unset and forces `gfx950` on SPUR; empty omits the arch suffix from the image tag |
| `AIC_ROCM_ARCH` | `gfx90a;gfx942;gfx950;gfx1151;gfx1201` | `.slurm/run-build-distribute.sh` | Arch list a Slurm build compiles for and a Slurm test expects (`;`-separated). Separate from `ROCM_ARCH` on purpose: the default covers every supported arch, so one image runs on any of them |
| `AIC_FAST_ARCH` | `gfx950` | `Makefile` | The single arch of the `-fast` make targets (`dist-build-fast`, `smoke-test-fast`, ...) |
| `ROCM_BASE_IMAGE` | `rocm/dev-ubuntu-26.04:${ROCM_VERSION}-full` | `docker/docker-bake.hcl` | Full ROCm development image the source builds run on |
| `ROCM_RUNTIME_IMAGE` | `ubuntu:26.04` | `docker/docker-bake.hcl` | Image the shipped images start from |
<!-- END generated:rocm -->

### Version pins

<!-- BEGIN generated:pins -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `PYTORCH_REF` | `abecaa87238d2d43d2641a22d89a51de185779b2` | `docker/docker-bake.hcl` | `ROCm/pytorch` commit built from source. A SHA, because the 2.15 series has no tag |
| `PYTORCH_SERIES` | `2.15` | `docker/docker-bake.hcl` | Display only: the torch series in the image tag, README and release notes, and the CPU torch the emulate builds install. Bump with `PYTORCH_REF` |
| `TORCHVISION_REF` | `v0.29.2` | `docker/docker-bake.hcl` | Ref of torchvision built against that PyTorch |
| `VLLM_REF` | `v0.34.0` | `docker/docker-bake.hcl` | vLLM ref |
| `LLM_EMU_REF` | `b71c602` | `docker/docker-bake.hcl` | llm-emu ref (the emulation plugin) |
| `AITER_REF` | `v0.1.29.post1` | `docker/docker-bake.hcl` | ROCm AITER ref installed into every image; override only together with a validated `FLASH_ATTN_REF` |
| `FLASH_ATTN_REF` | `0e60e394` | `docker/docker-bake.hcl` | FlashAttention ref, built for the CDNA (`gfx9*`) arches only |
| `LMCACHE_REF` | `v0.5.8` | `docker/docker-bake.hcl` | LMCache ref |
| `NIXL_REF` | `v1.5.2` | `docker/docker-bake.hcl` | NIXL ref |
| `CUPY_VERSION` | `14.2.0` | `docker/docker-bake.hcl` | `cupy-rocm-7-0` release installed into `aic-lmcache` and `aic-vllm` |
| `UCX_REF` | `v1.22.0` | `docker/docker-bake.hcl` | UCX ref (NIXL's transport) source-built into the base image |
| `HSA_SNOOP_REF` | `aba6a306` | `docker/docker-bake.hcl` | hsa-snoop ref |
| `CLR_REF` | `release/therock-10.2` | `docker/docker-bake.hcl` | `ROCm/rocm-systems` ref the patched `libamdhip64` in the base image is built from; bump it together with `ROCM_VERSION` |
| `TRITON_REF` | `release/internal/3.8.x` | `docker/docker-bake.hcl` | Triton ref source-built into the base image |
| `NVME_EXPORTER_VERSION` | `3.0.0` | `docker/docker-bake.hcl` | `nvme_exporter` release packaged into the NVMe exporter image |
| `RDMA_EXPORTER_VERSION` | `0.7.5` | `docker/docker-bake.hcl` | `rdma_exporter` release packaged into the RDMA exporter image |
| `KVBENCH_REF` | `177cf4e7c46ece8e03a701f89b60cd3a329f45fd` | `docker/docker-bake.hcl` | KVBench commit |
| `KVBENCH_LMCACHE_VERSION` | `0.5.6` | `docker/docker-bake.hcl` | LMCache release (from `PyPI`) installed into the KVBench image. Not `LMCACHE_REF`: KVBench pins its own compatible range |
| `AIPERF_VERSION` | `0.13.0` | `docker/docker-bake.hcl` | `aiperf` release installed into the `aic-client` image, in its own virtualenv; `latest` takes whatever `PyPI` serves on the day of the build |
| `AGENT_KV_REF` | `19fcc86be6e267839889567ce5d20dd0fed6bbaf` | `docker/docker-bake.hcl` | agent-kv commit installed into the `aic-client` image. A SHA, because agent-kv has no tags |
<!-- END generated:pins -->

### Source repositories

<!-- BEGIN generated:repositories -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `PYTORCH_GIT_URL` | `https://github.com/ROCm/pytorch.git` | `docker/docker-bake.hcl` | PyTorch source repository; override to build from a fork |
| `TORCHVISION_GIT_URL` | `https://github.com/pytorch/vision.git` | `docker/docker-bake.hcl` | Source repository of torchvision |
| `VLLM_GIT_URL` | `https://github.com/vllm-project/vllm.git` | `docker/docker-bake.hcl` | vLLM source repository |
| `LLM_EMU_GIT_URL` | `https://github.com/AKafakA/llm-emu.git` | `docker/docker-bake.hcl` | llm-emu source repository |
| `AITER_GIT_URL` | `https://github.com/ROCm/aiter.git` | `docker/docker-bake.hcl` | AITER source repository |
| `FLASH_ATTN_GIT_URL` | `https://github.com/Dao-AILab/flash-attention.git` | `docker/docker-bake.hcl` | FlashAttention source repository |
| `LMCACHE_GIT_URL` | `https://github.com/LMCache/LMCache.git` | `docker/docker-bake.hcl` | LMCache source repository |
| `NIXL_GIT_URL` | `https://github.com/ai-dynamo/nixl.git` | `docker/docker-bake.hcl` | NIXL source repository |
| `UCX_GIT_URL` | `https://github.com/openucx/ucx.git` | `docker/docker-bake.hcl` | UCX source repository |
| `HSA_SNOOP_GIT_URL` | `https://github.com/sbates130272/hsa-snoop.git` | `docker/docker-bake.hcl` | hsa-snoop source repository |
| `CLR_GIT_URL` | `https://github.com/ROCm/rocm-systems.git` | `docker/docker-bake.hcl` | Source repository of the patched `libamdhip64` |
| `TRITON_GIT_URL` | `https://github.com/ROCm/triton.git` | `docker/docker-bake.hcl` | Source repository of Triton |
| `KVBENCH_GIT_URL` | `https://github.com/wvaske/llm-kv-passthrough.git` | `docker/docker-bake.hcl` | KVBench source repository |
| `AGENT_KV_GIT_URL` | `https://github.com/sbates130272/agent-kv` | `docker/docker-bake.hcl` | agent-kv source repository; the build fetches the commit as a tarball |
<!-- END generated:repositories -->

### Build controls and layer cache

<!-- BEGIN generated:build -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `BUILD_JOBS` | (empty) | `docker/docker-bake.hcl` | Cap on parallel compile jobs. Empty means `nproc`; a Slurm build defaults it to `AIC_BUILD_CPUS` |
| `BUILD_PROGRESS` | `auto` | `mk/local.mk` | `docker buildx bake --progress` mode for `make build` (`auto`, `plain`, `tty`) |
| `BAKE_TARGETS` | `default` | `mk/local.mk` | Bake targets `make build` builds |
| `BAKE_OUTPUT_FLAG` | `--load` | `mk/local.mk` | Output flag `make build` passes to bake (`--load`, `--push`) |
| `WHEELS_DIR` | `wheels-out`; `$(REPO_ROOT)/wheels-out` (`mk/local.mk`) | `docker/docker-bake.hcl` | Where the `wheels` bake group writes the pip wheels, one directory per wheel target; `make wheels` flattens them |
| `VLLM_TARGET_DEVICE` | `rocm` | `docker/docker-bake.hcl` | vLLM build target: `rocm` compiles the GPU kernels, `empty` compiles none (the emulation image) |
| `AIC_UCX_FAST` | (empty) | `docker/docker-bake.hcl` | `1` builds a minimal UCX; `make dist-build-fast` sets it |
| `BLAS_CACHE_BUST` | (empty) | `docker/docker-bake.hcl` | Cache key of the ROCm runtime-package layer in the base image. Change it to reinstall those packages, e.g. `BLAS_CACHE_BUST=$(date +%s) make build` |
| `AITER_PREBUILD_KERNELS` | `0` | `docker/docker-bake.hcl` | Compile the AITER kernels into the vLLM image at build time instead of on first use. `1` most, `2` no backward/tune, `3` `fmha_v3` only. Single-arch builds only |
| `TLS_CERT` | (empty) | `docker/docker-bake.hcl` | Corporate CA certificate passed to the build as a BuildKit secret, never stored in an image. A Slurm build falls back to `$HOME/certs/zscaler-ca.crt` when that file exists |
| `AIC_BUILDX_BUILDER` | `aic-local` | `Makefile` | Name of the `docker-container` buildx builder used by `make build` and the Slurm build; ignored with the `docker` driver |
| `AIC_CACHE_REF` | (empty) | `docker/docker-bake.hcl` | Registry ref for a shared BuildKit layer cache, stored as `<ref>-<arch>-<image>`. Takes precedence over `AIC_CACHE_GHA_SCOPE` and `AIC_CACHE_DIR`; needs `docker login` |
| `AIC_CACHE_GHA_SCOPE` | (empty) | `docker/docker-bake.hcl` | GitHub Actions cache scope prefix, for small GitHub-hosted builds only (the Actions cache is far smaller than this stack's layers) |
| `AIC_CACHE_DIR` | (empty); `$(AIC_SHARED_NFS)/$(USER)/buildcache` (`Makefile`, SPUR) | `docker/docker-bake.hcl` | Directory for a file-based BuildKit layer cache, one directory per arch. With all three cache variables empty the build uses no cache |
| `AIC_CACHE_MODE` | `max` | `docker/docker-bake.hcl` | BuildKit `--cache-to` mode: `min` or `max` |
<!-- END generated:build -->

### Slurm build and distribution

`.slurm/run-build-distribute.sh` (the `make dist-*` targets). See
[SLURM_SPUR.md](SLURM_SPUR.md).

<!-- BEGIN generated:slurm-build -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `AIC_IMAGE_DIR` | `${AIC_SHARED_NFS:-/shared_nfs}/rocm-aic/images`; `$(AIC_SHARED_NFS)/rocm-aic/images` (`Makefile`, SPUR); `/scratch/${USER}/images` (another branch); `${AIC_SHARED_NFS}/rocm-aic/images` (`.slurm/ext-hit-test.sbatch`) | `.slurm/run-build-distribute.sh` | Shared directory the built image artifacts (tarballs or registry refs) are stored in and loaded from. The first default applies on SPUR, the alternate elsewhere |
| `AIC_TARGETS` | (empty) | `.slurm/run-build-distribute.sh` | Comma-separated nodes the `load` and `all` commands load the images onto |
| `AIC_DIST_BACKEND` | `registry`; `tarball` (another branch) | `.slurm/run-build-distribute.sh` | How a build hands its images to the jobs that run them: `tarball` (`docker save` into `AIC_IMAGE_DIR`) or `registry` (a registry's storage on shared NFS plus a digest ref, so layers are stored once and a node fetches only what it lacks). `registry` needs `AIC_BUILDX_DRIVER=docker` and is the default on SPUR with that driver; `tarball` elsewhere. `make cache-gc` deletes the images no job has used in `AIC_CACHE_GC_RETAIN_DAYS` |
| `AIC_REGISTRY_IMAGE` | `registry:3.1.2` | `.slurm/image-dist-lib.sh` | Image of the registry the jobs start for `AIC_DIST_BACKEND=registry` and `AIC_CACHE_BACKEND=registry`. A node without it loads `<storage>/aic-images/<image>.tar`, which the first job to pull it writes |
| `AIC_CACHE_BACKEND` | `registry`; `local` (another branch) | `.slurm/run-build-distribute.sh` | Format `AIC_CACHE_DIR` is kept in: `local` (a BuildKit local cache) or `registry` (a registry's storage under `<dir>/registry`, served by a container the build job starts and removes). `registry` needs `AIC_BUILDX_DRIVER=docker` and is the default on SPUR with that driver; `local` elsewhere. `make cache-gc` deletes the cache no job has used in `AIC_CACHE_GC_RETAIN_DAYS` |
| `AIC_CACHE_GC_RETAIN_DAYS` | `7` | `.slurm/run-build-distribute.sh` | `make cache-gc` keeps every cache manifest and image a job pushed, pulled or read in this many days, and any image a `.ref` under `AIC_CACHE_GC_REF_DIRS` names; it deletes nothing until usage has been recorded for this long |
| `AIC_CACHE_GC_DRY_RUN` | `0` | `.slurm/run-build-distribute.sh` | `1` makes `make cache-gc` report what it would delete and delete nothing |
| `AIC_CACHE_GC_LOCK_WAIT` | `3600` | `.slurm/run-build-distribute.sh` | Seconds `make cache-gc` waits for the jobs serving the registry storage to finish; must stay under `AIC_CACHE_GC_TIME` |
| `AIC_CACHE_GC_TIME` | `02:00:00` | `.slurm/run-build-distribute.sh` | Slurm time limit of the `make cache-gc` job, including `AIC_CACHE_GC_LOCK_WAIT` |
| `AIC_CACHE_GC_REF_DIRS` | `${AIC_IMAGE_DIR}` | `.slurm/run-build-distribute.sh` | Colon-separated trees `make cache-gc` searches for `.ref` files; an image one names is kept however old |
| `AIC_CACHE_GC_IDLE_JOBS` | `aic-*` (only when unset) | `.slurm/run-build-distribute.sh` | Glob of your Slurm job names: `make cache-gc` skips its run while one of them is running, since a checkout that predates the storage lock does not take it. Empty skips the check |
| `AIC_REGISTRY_LOCK_WAIT` | `3600` | `.slurm/image-dist-lib.sh` | Seconds a job waits for `make cache-gc` to finish with the registry storage before it fails |
| `AIC_CACHE_RESET` | `1` | `.slurm/run-build-distribute.sh` | `1` passes `reset=true` to a local `--cache-to`, so superseded blobs are dropped; needs buildx 0.35.0 or newer |
| `AIC_CACHE_INSECURE` | (empty) | `.slurm/run-build-distribute.sh` | `1` when `AIC_CACHE_REF` presents a certificate the `docker-container` builder does not trust: it is told to skip verification |
| `AIC_BUILDX_DRIVER` | `docker`; `docker-container` (another branch) | `.slurm/run-build-distribute.sh` | `docker-container` (a builder the script creates on the build node) or `docker` (BuildKit inside the daemon, which needs the `containerd` image store). With `docker` no builder is created or pruned and `docker/buildkitd.toml` cannot be applied, so step logs are clipped. `docker` is the default on SPUR, where the privileged builder container is denied; `docker-container` elsewhere |
| `AIC_BUILDX_PRUNE` | `0` | `.slurm/run-build-distribute.sh` | `1` prunes the builder's node-local cache after a `bake`; the next build compiles everything again |
| `AIC_BUILD_PARTITION` | `${AIC_SPUR_PARTITION:-amd-spur-v2}`; `defq` (another branch) | `.slurm/run-build-distribute.sh` | Slurm partition for the build and load jobs. `AIC_SPUR_PARTITION` on SPUR, `defq` elsewhere |
| `AIC_BUILD_CONSTRAINT` | (empty); `CPUONLY` (another branch) | `.slurm/run-build-distribute.sh` | Slurm `-C` expression for the build node, used when `AIC_BUILD_NODE` is unset. Empty on SPUR, where nodes carry no feature labels; `CPUONLY` elsewhere |
| `AIC_BUILD_NODE` | (empty) | `.slurm/run-build-distribute.sh` | Pin the build node with `--nodelist`; overrides `AIC_BUILD_CONSTRAINT` |
| `AIC_BUILD_EXCLUDE_NODES` | (empty) | `.slurm/run-build-distribute.sh` | Comma-separated nodes excluded from every submit. Ignored when a node is pinned |
| `AIC_BUILD_LOCAL` | (empty) | `.slurm/run-build-distribute.sh` | `1` builds on this host, with no Slurm job |
| `AIC_BUILD_CPUS` | `32` | `.slurm/run-build-distribute.sh` | `--cpus-per-task` of the build job |
| `AIC_BUILD_MEM` | (empty) | `.slurm/run-build-distribute.sh` | `--mem` of the build job; empty leaves it to the partition default |
| `AIC_BUILD_GPUS` | `1` | `.slurm/run-build-distribute.sh` | `--gpus` of the build job. SPUR refuses a submit with no GPU request, so it defaults to `1` there; unset elsewhere |
| `AIC_BUILD_TIME` | `04:00:00` | `.slurm/run-build-distribute.sh` | Build job time limit |
| `AIC_LOAD_TIME` | `00:30:00` | `.slurm/run-build-distribute.sh` | Per-node load job time limit |
| `AIC_TMP_DIR` | (empty) | `.slurm/run-build-distribute.sh` | Build-job `TMPDIR` and OCI scratch. Empty picks node-local `/tmp/aic-scratch.<job>`, then `$HOME/.tmp-rocm-aic-cicd`; never shared NFS |
| `AIC_COMPRESS` | (empty) | `.slurm/run-build-distribute.sh` | Tarball compression: `zstd`, `gzip` or `none`. Empty picks `zstd` when available, else `gzip` |
| `AIC_BUILD_TARGET` | (empty) | `.slurm/run-build-distribute.sh` | Dockerfile stage the single-image `build` command stops at (`emulate` for the emulation image) |
| `AIC_BUILD_DOCKERFILE` | `lmcache/Dockerfile` | `.slurm/run-build-distribute.sh` | Dockerfile (relative to `docker/`) the single-image `build` command builds |
| `AIC_BUILD_CONTEXT_BASE` | (empty) | `.slurm/run-build-distribute.sh` | Pre-built base image handed to that build as its `base` stage |
| `AIC_BUILD_BASE_FIRST` | (empty) | `.slurm/run-build-distribute.sh` | `1` builds the base image first, in the same job, instead |
| `AIC_BUILD_CONTEXT_BASE_WHEELS` | (empty) | `.slurm/run-build-distribute.sh` | Directory of exported base wheels handed to that build as its `base-wheels` context |
| `AIC_BUILD_CONTEXT_LMCACHE_WHEELS` | (empty) | `.slurm/run-build-distribute.sh` | Directory of exported LMCache wheels handed to that build as its `lmcache-wheels` context |
| `AIC_VLLM_TARGET_DEVICE` | (empty) | `.slurm/run-build-distribute.sh` | Slurm name for bake's `VLLM_TARGET_DEVICE` in the single-image build |
| `AIC_EMULATE_VLLM_DEVICE` | (empty) | `.slurm/run-build-distribute.sh` | `VLLM_TARGET_DEVICE` for `build-emulate`; empty takes the emulate target's `empty` |
| `AIC_AITER_PREBUILD_KERNELS` | (empty) | `.slurm/run-build-distribute.sh` | Slurm name for bake's `AITER_PREBUILD_KERNELS`; needs a single-arch `AIC_ROCM_ARCH` |
| `AIC_WHEEL_DIR` | `${AIC_IMAGE_DIR}/wheels` | `.slurm/run-build-distribute.sh` | Where the `wheels` command exports the pip wheels |
| `AIC_VLLM_PUSH_REF` | (empty) | `.slurm/run-build-distribute.sh` | Registry-qualified ref `make dist-push` pushes the vLLM image to. Both push refs are required |
| `AIC_LMCACHE_PUSH_REF` | (empty) | `.slurm/run-build-distribute.sh` | Registry-qualified ref `make dist-push` pushes the LMCache image to |
| `AIC_FORCE_LOAD` | `0` | `.slurm/run-build-distribute.sh` | `1` reloads an image from its tarball even when the node's copy is current. By default a node reloads only when the tarball is newer than what it last loaded |
| `AIC_REQUIRE_FRESH_TARBALL` | `0` | `.slurm/run-build-distribute.sh` | `1` fails a build whose tarball was not rewritten, instead of warning |
| `AIC_JOB_LOG_TAIL` | `200` | `.slurm/run-build-distribute.sh` | Lines of the job log printed when a submitted job fails |
| `AIC_SLURM_ACCOUNT` | (empty) | `.slurm/run-build-distribute.sh` | Slurm `--account` for every submit; not needed on SPUR |
<!-- END generated:slurm-build -->

## Cluster and site

<!-- BEGIN generated:site -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `AIC_CLUSTER` | `$(firstword $(foreach _t,$(SLURM_CLUSTER_NAME) $(subst -, ,$(shell hostname -s 2>/dev/null)), $(if $(wildcard $(REPO_ROOT)/.env.$(_t)),$(_t))))`; `$(_aic_detect_cluster)` (`.slurm/run-build-distribute.sh`) | `Makefile` | Selects the per-cluster environment file `.env.<cluster>` in the repo root. Auto-detected: the first dash-separated token of the host name with a matching file |
| `AIC_ENV_FILE` | `$(if $(AIC_CLUSTER),$(REPO_ROOT)/.env.$(AIC_CLUSTER))`; `${AIC_CLUSTER:+${AIC_DAY_DIR}/.env.${AIC_CLUSTER}}` (`.slurm/run-build-distribute.sh`) | `Makefile` | Path of that file, overriding the `AIC_CLUSTER` lookup. Plain unquoted `KEY=value` lines |
| `AIC_SPUR_CLUSTER` | `0`; `1` (`.slurm/ext-hit-test.sbatch`) | `Makefile` | `1` on a SPUR controller: selects the SPUR site defaults shown in these tables and the SPUR submit path. SPUR's `sbatch` has no `--parsable`, `--wait` or `--overcommit`, so jobs are submitted from a file and watched with `squeue` |
| `AIC_SPUR_PARTITION` | `amd-spur-v2` (SPUR) | `Makefile` | Partition for every SPUR submit. Use the variable, not the literal name |
| `AIC_SHARED_NFS` | `/shared_nfs` (SPUR) | `Makefile` | Root of the shared NFS on SPUR, under which the image, cache and Hugging Face directories default |
| `AIC_SPUR_RPC_RETRIES` | `30` | `.slurm/run-build-distribute.sh` | Consecutive `squeue` failures tolerated while watching a job (10 s apart) before giving up |
| `AIC_CI_LIB_DIR` | `/usr/local/lib/aic-ci` | `Makefile` | Where `make install-ci-scripts` installs the self-hosted runner scripts |
| `AIC_CI_STORAGE_ROOT` | `$HOME/Projects/rocm-aic-ci` | `.github/scripts/runners/spur-ci-common.sh` | Root of the CI checkouts on the SPUR login node |
| `AIC_CI_ACTIVE_JOB_FILE` | (empty) | `.slurm/run-build-distribute.sh` | CI runners: file the id of the job being watched is written to, so a cancelled workflow can cancel it |
<!-- END generated:site -->

## Stack runtime

### GPU and host storage

<!-- BEGIN generated:gpu-storage -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `GPU` | `0` | `docker/docker-compose.yml` | Fallback GPU index when no scheduler allocation is resolved; also names the vLLM container (`<stack>-vllm-gpu<GPU>`) |
| `AIC_ROCR_VISIBLE` | `${GPU:-0}` | `docker/docker-compose.yml` | `ROCR_VISIBLE_DEVICES` for every container with `/dev/kfd`: absolute host GPU ids (`3` or `2,5`). The Slurm jobs resolve it: on standard Slurm from `ROCR_VISIBLE_DEVICES`, then `HIP_VISIBLE_DEVICES`, then `CUDA_VISIBLE_DEVICES`; on SPUR from the controller allocation, and an allocation that cannot be resolved fails the job |
| `AIC_HIP_VISIBLE` | `0` | `docker/docker-compose.yml` | `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`: indices relative to `AIC_ROCR_VISIBLE` (`0` or `0,1`). Set the two together |
| `NVME_DATA` | `/mnt/lmcache-nvme`; `/tmp/lmcache-emulate-nvme` (`lmcache-emulator`); `/mnt/m2m_nobackup/aic-cliff/nvme` (`Makefile`, SPUR); `${AIC_LOCAL_ROOT}/nvme` (`.slurm/run-cliff.sbatch`); `/mnt/m2m_nobackup/aic-cliff-exthit/nvme` (`.slurm/ext-hit-test.sbatch`) | `docker/docker-compose.yml` | Host path of the local NVMe pool (L2a), mounted at `/data/nvme`. The cliff job picks its own under `AIC_LOCAL_ROOT` |
| `NFS_DATA` | `/mnt/lmcache-nfs`; `/tmp/lmcache-emulate-nfs` (`lmcache-emulator`); `${AIC_LOCAL_ROOT}/nfs` (`.slurm/run-cliff.sbatch`) | `docker/docker-compose.yml` | Host path of the NFS-over-RDMA pool (L2b), mounted at `/data/nfs` |
| `GDS_SLAB_DATA` | (empty); `/tmp/gds-slab-unused` (`lmcache`); `/mnt/m2m_nobackup/aic-cliff/slab` (`Makefile`, SPUR); `${AIC_LOCAL_ROOT}/slab` (`.slurm/run-cliff.sbatch`) | `docker/docker-compose.yml` | Host path of the hipFile slab file; required with `GDS_MODE=1` |
| `GDS_MODE` | (empty); `0` (`x-lmcache-command`) | `docker/docker-compose.yml` | `1` uses the hipFile NVMe slab as L1 instead of DRAM (`make up-gds-l1`) |
| `AIC_SHM_SIZE` | `64gb` | `docker/docker-compose.yml` | Size of the private `/dev/shm` shared by LMCache and vLLM; raise it when the L1 request is larger |
<!-- END generated:gpu-storage -->

### Hugging Face and logging

<!-- BEGIN generated:hf-logging -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `HF_TOKEN` | (empty) | `docker/docker-compose.yml` | Hugging Face access token. Required to download gated models; not needed for an offline cache |
| `HF_TOKEN_FILE` | (empty) | `Makefile` | File holding the token, read by make when `HF_TOKEN` is unset |
| `HF_HOME` | `~/.cache/huggingface`; `$(HOME)/.cache/huggingface` (`Makefile`); `$(AIC_SHARED_NFS)/huggingface` (`Makefile`, SPUR); `${AIC_IMAGE_DIR}/tiny-hf` (`.slurm/run-build-distribute.sh`); `/scratch/${USER}/vllm-lmcache-hipfile/hf` (`.slurm/run-cliff.sbatch`); `${AIC_SHARED_NFS}/huggingface` (`.slurm/ext-hit-test.sbatch`); `${AIC_SHARED_NFS:-/shared_nfs}/huggingface` (`.slurm/test-nightly-wheels.sbatch`); `${RUNNER_TEMP:-/tmp}/hf` (`.github/scripts/run-vllm-emul-perf.sh`); `${HOME}/.cache/huggingface` (`benchmarks/agentx/run_agentx.sh`) | `docker/docker-compose.yml` | Host Hugging Face cache directory mounted into the containers. `make` uses `/scratch/models` when that exists; each Slurm job has its own default |
| `HF_HUB_OFFLINE` | `0` | `docker/docker-compose.yml` | `1` blocks Hugging Face Hub network access |
| `TRANSFORMERS_OFFLINE` | `0` | `docker/docker-compose.yml` | `1` puts Transformers in offline mode |
| `LOG` | `../logs`; `$(CURDIR)/logs` (`Makefile`); `${REPO_ROOT}/logs` (`benchmarks/agentx/run_agentx.sh`) | `docker/docker-compose.yml` | Host log directory |
| `TZ` | `America/Edmonton` | `docker/docker-compose.yml` | Time zone of every container |
<!-- END generated:hf-logging -->

### LMCache server and L2 backend

<!-- BEGIN generated:lmcache -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `LMCACHE_PORT` | `6555` | `docker/docker-compose.yml` | ZMQ listen port of the LMCache MP server |
| `LMCACHE_L1_SIZE_GB` | `20` | `docker/docker-compose.yml` | L1 cap in GiB: DRAM in the standard mode, slab size in GDS mode |
| `AIC_L2_BACKEND` | `nixl`; `nixl_posix` (`lmcache-emulator`); `nixl_posix` (`Makefile`, SPUR) | `docker/docker-compose.yml` | L2 backend: `nixl` (AIS_MT NVMe plus POSIX NFS), `nixl_posix`, `nixl_posix_dynamic`, `local_disk` (LMCache's own disk backend through a mounted config) or `none`. `nixl_posix` is the SPUR and emulation default |
| `AIC_NIXL_BACKEND` | `AIS_MT` | `docker/docker-compose.yml` | NIXL plugin of the `nixl` backend: `AIS_MT` (hipFile, P2PDMA) or `POSIX` |
| `AIC_NIXL_USE_DIRECT_IO` | `true` | `docker/docker-compose.yml` | `O_DIRECT` for the AIS_MT NVMe pool. The POSIX plugin always uses buffered I/O |
| `LMCACHE_NVME_POOL` | `4096` | `docker/docker-compose.yml` | NIXL pool slots of the NVMe adapter |
| `LMCACHE_NVME_SLOT_SIZE` | `268435456` | `docker/docker-compose.yml` | File size per NVMe pool slot, bytes (256 MiB) |
| `LMCACHE_NFS_POOL` | `1024` | `docker/docker-compose.yml` | NIXL pool slots of the NFS adapter |
| `LMCACHE_NIXL_POSIX_SLOT_SIZE` | `16777216` | `docker/docker-compose.yml` | Size of each POSIX pool file, bytes (16 MiB). L2 capacity is pool x this size; a multiple of 4 KiB |
| `LMCACHE_NIXL_POSIX_POOL` | `32768` | `docker/docker-compose.yml` | Pool slots of the POSIX backend |
| `LMCACHE_NIXL_DYNAMIC_MAX_GB` | `100` | `docker/docker-compose.yml` | NVMe capacity cap in GB for `nixl_posix_dynamic` |
| `LMCACHE_NIXL_DYNAMIC_DIRECT_IO` | `false` | `docker/docker-compose.yml` | `O_DIRECT` for `nixl_posix_dynamic` |
| `LMCACHE_MAX_GPU_WORKERS` | `4` | `docker/docker-compose.yml` | `--max-gpu-workers`, the parallel store and retrieve workers. `1` puts every client on one thread and backs stores up under concurrent load; use it only for a single-client run, to save GPU memory |
| `LMCACHE_WORKER_REAP_TIMEOUT` | `3600` | `docker/docker-compose.yml` | `--worker-reap-timeout-seconds` |
| `LMCACHE_L2_PREFETCH_MAX_IN_FLIGHT` | `4`; `32` (`.slurm/run-cliff.sbatch`) | `docker/docker-compose.yml` | `--l2-prefetch-max-in-flight`, concurrent L2 load tasks. The cliff job raises it |
| `LMCACHE_CHUNK_SIZE` | `256` | `docker/docker-compose.yml` | Tokens per KV chunk; the MP server and the coordinator must agree |
| `LMCACHE_CONFIG_FILE_HOST` | `/dev/null` | `docker/docker-compose.yml` | Host path of an LMCache config file to mount (`local_disk` and advanced tuning) |
| `LMCACHE_CONFIG_FILE` | (empty) | `docker/docker-compose.yml` | In-container path of that config; set only when one is mounted |
| `NIXL_TELEMETRY_ENABLE` | `y` (only when unset) | `docker/docker-compose.yml` | NIXL's own Prometheus exporter on `:19090`. Set it empty to disable |
| `NIXL_TELEMETRY_EXPORTER` | `prometheus` | `docker/docker-compose.yml` | NIXL telemetry exporter type |
<!-- END generated:lmcache -->

### LMCache coordinator

<!-- BEGIN generated:coordinator -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `LMCACHE_COORDINATOR_URL` | `http://aic-lmcache-coordinator:9300` (only when unset) | `docker/docker-compose.yml` | Base URL the MP server registers and heartbeats against. Set it empty to run without a coordinator |
| `LMCACHE_COORDINATOR_INSTANCE_TIMEOUT` | `120` | `docker/docker-compose.yml` | Seconds without a heartbeat before an instance is evicted |
| `LMCACHE_COORDINATOR_HEALTH_INTERVAL` | `10` | `docker/docker-compose.yml` | Seconds between heartbeat sweeps |
| `LMCACHE_COORDINATOR_EVICTION_INTERVAL` | `5` | `docker/docker-compose.yml` | Seconds between L2 eviction checks |
| `LMCACHE_COORDINATOR_KEEPALIVE` | `65` | `docker/docker-compose.yml` | HTTP keep-alive timeout, seconds |
<!-- END generated:coordinator -->

### vLLM serving

<!-- BEGIN generated:vllm -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `VLLM_MODEL` | `openai/gpt-oss-120b`; `Qwen/Qwen3-8B` (`vllm-emulator`); `Qwen/Qwen2.5-3B-Instruct` (`.slurm/ext-hit-test.sbatch`); `Qwen/Qwen2.5-14B-Instruct-1M` (`.slurm/test-nightly-wheels.sbatch`) | `docker/docker-compose.yml` | Model to serve. The cliff job picks one by arch when it is empty (`AIC_MODEL_BIG`, `AIC_MODEL_SMALL`) |
| `TENSOR_PARALLEL_SIZE` | `1` | `docker/docker-compose.yml` | `--tensor-parallel-size` |
| `VLM_GPU_MEMORY_UTILIZATION` | `0.90`; `0.85` (`vllm-emulator`) | `docker/docker-compose.yml` | `--gpu-memory-utilization` |
| `VLM_MAX_MODEL_LEN` | `32768`; `4096` (`vllm-emulator`) | `docker/docker-compose.yml` | `--max-model-len` |
| `VLM_MAX_NUM_BATCHED_TOKENS` | `4096` | `docker/docker-compose.yml` | `--max-num-batched-tokens` |
| `VLM_BLOCK_SIZE` | `64` | `docker/docker-compose.yml` | `--block-size`; match it to the model and connector cache geometry |
| `VLM_LOAD_FORMAT` | `fastsafetensors` | `docker/docker-compose.yml` | `--load-format`. `fastsafetensors` is CDNA-only; use `auto` on RDNA |
| `VLM_ATTENTION_BACKEND` | `TRITON_ATTN` | `docker/docker-compose.yml` | `--attention-backend`. `TRITON_ATTN` is the one that supports KV connectors |
| `VLM_KV_CACHE_DTYPE` | `auto` | `docker/docker-compose.yml` | `--kv-cache-dtype`. `auto` stores KV in the model dtype; `fp8` roughly doubles KV capacity where the arch has native `fp8`, but gave repeated-token glitches with Qwen2.5-3B on gfx1201 |
| `VLM_KV_CACHE_METRICS_SAMPLE` | `0.05` | `docker/docker-compose.yml` | Fraction of KV blocks sampled for the residency histograms |
| `KV_TRANSFER_ARG` | (empty); `--kv-transfer-config '$(_MP_CONNECTOR_JSON)'` (`Makefile`) | `docker/docker-compose.yml` | vLLM `--kv-transfer-config` argument. `make` sets the `LMCacheMPConnector` JSON; empty is a plain vLLM (the cliff `vram` baseline). Single-quote the JSON |
| `LMCACHE_LAZY_OFFLOAD` | `true` | `Makefile` | `lmcache.mp.lazy_offload` in the connector JSON `make` builds. `false` stores every block as it is computed, which `make agent-kv-nvme-sweep` needs: a `/reset_prefix_cache` drops deferred stores |
| `LMCACHE_USE_VMM_API` | `false` | `Makefile` | `lmcache.mp.use_vmm_api` in the connector JSON `make` builds. `true` uses the ROCm VMM fd-export path; requires `patches/vllm/02-vllm-rocm-extensible-kv-fd-exporter.patch` in the build |
| `VLLM_EXTRA_ARGS` | (empty) | `docker/docker-compose.yml` | Extra `vllm serve` arguments appended verbatim; single-quote embedded JSON |
| `VLLM_LOGGING_LEVEL` | `INFO`; `WARNING` (`.slurm/run-cliff.sbatch`) | `docker/docker-compose.yml` | vLLM log level |
| `VLLM_SERVER_DEV_MODE` | `1` | `docker/docker-compose.yml` | `1` enables vLLM's development endpoints, which the tests use to reset the GPU prefix cache |
| `AIC_ROCM_USE_AITER` | `1` | `docker/docker-compose.yml` | `VLLM_ROCM_USE_AITER`: the AITER unified attention a KV connector needs. `0` for arches without AITER support |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:False` | `docker/docker-compose.yml` | Passed through to vLLM. A no-op on ROCm; see `PYTORCH_HIP_ALLOC_CONF` |
| `PYTORCH_HIP_ALLOC_CONF` | `expandable_segments:False` | `docker/docker-compose.yml` | PyTorch HIP allocator configuration for vLLM |
| `VLLM_IPC_MODE` | `service:lmcache` | `docker/docker-compose.yml` | IPC namespace of the `vllm` service. It joins LMCache's; to run vLLM alone start the `vllm-only` service rather than overriding this |
| `VLLM_PID_MODE` | `service:lmcache` (only when unset) | `docker/docker-compose.yml` | PID namespace of the `vllm` service. Empty means a private namespace |
| `VLLM_NETWORK_MODE` | `container:${AIC_STACK:-aic}-lmcache` | `docker/docker-compose.yml` | Network namespace of the `vllm` service. ROCR passes IPC handles over an abstract socket, so vLLM and LMCache must share one |
| `VLM_READY_RETRIES` | `60` | `Makefile` | `make vllm-reset-test`: readiness polls, 5 s apart |
| `VLM_FAIL_LOG_LINES` | `250` | `Makefile` | `make vllm-reset-test`: container log lines printed when the vLLM container exits before becoming ready |
<!-- END generated:vllm -->

### Clients and published ports

Only Grafana publishes a host port; every other service is reached by
container name on the `aic` bridge, from the `aic-client` container (see
[Client image and `aic-bench`](#client-image-and-aic-bench)).

<!-- BEGIN generated:network -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `AIC_STACK` | `aic` | `docker/docker-compose.yml` | Stack name: the compose project, the bridge network and every container's prefix (`<stack>-lmcache`, `<stack>-vllm-gpu<GPU>`, ...), so stacks with different names share a host untouched. make and the scripts default it to `aic-<job id>` inside a Slurm job and elsewhere to `aic-` and eight hex digits of a random UUID, generated once per checkout and kept in `.aic-stack`. Plain `docker compose` falls back to `aic`. Set it to choose a name, and pass the same value to every later command that should reach it. On its network each container also answers to its fixed `aic-*` alias |
| `AIC_PUBLISH_ADDR` | `127.0.0.1` | `docker/docker-compose.yml` | Host address the published Grafana port binds; `0.0.0.0` exposes it beyond loopback |
| `AIC_GRAFANA_PORT` | (empty) | `docker/docker-compose.yml` | Pin the Grafana host port, the only published one. Unset, `make` picks the lowest free port from 3000 and keeps it across a down/up; empty, or compose run without `make`, lets Docker pick a free port. `make endpoints` prints the URL |
| `AIC_CLIENT_IMAGE` | `aic-client:latest`; `aic-client:${AIC_VERSION}` (`.slurm/run-build-distribute.sh`); `aic-client:$(<"${REPO}/VERSION")` (`.slurm/run-cliff.sbatch`) | `docker/docker-compose.yml` | Image of the `aic-client` container and of the throwaway AgentX containers. `make` sets it to the ref bake tags (`aic-client:<AIC_VERSION>`); compose builds it from `docker/client/Dockerfile` when it is missing |
| `AIC_FETCH_IMAGE` | `curlimages/curl:8.23.0` | `scripts/aic-endpoints.sh` | Image `aic_fetch` runs `curl` from; scripts set it to an AIC image already on the node |
| `AIC_FETCH_TIMEOUT` | `60` | `scripts/aic-endpoints.sh` | `aic_fetch` request timeout, seconds |
| `AIC_NETWORK` | `${AIC_STACK}` | `benchmarks/agentx/run_agentx.sh` | Docker network the AgentX client joins; defaults to the stack's (`AIC_STACK`). `aic_fetch` and the endpoint helpers always use the stack's |
<!-- END generated:network -->

### Monitoring

See [METRICS_TELEMETRY.md](METRICS_TELEMETRY.md).

<!-- BEGIN generated:monitoring -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `AIC_MONITORING` | `1` | `monitoring/monitoring-lib.sh` | Slurm jobs: `1` starts the Prometheus sidecar, `0` skips it |
| `AIC_EXPORTERS` | `1`; `safe` (`Makefile`, SPUR) | `monitoring/monitoring-lib.sh` | Slurm jobs: `1` also starts the containerized exporter fleet, `0` is Prometheus only, `safe` is the subset SPUR permits. `make` exports `safe` on SPUR; the local make targets pick exporters with compose profiles instead |
| `AIC_METRICS_DIR` | `../logs/prometheus`; `$(CURDIR)/logs/prometheus` (`Makefile`); `${REPO}/logs/${SLURM_JOB_ID:-manual}/prometheus` (`.slurm/run-cliff.sbatch`) | `docker/docker-compose.yml` | Prometheus TSDB directory. The cliff job writes it under its own log directory |
| `AIC_PROM_IMAGE` | `prom/prometheus:v3.14.0` | `docker/docker-compose.yml` | Prometheus image |
| `AIC_PROM_RETENTION` | `90d` | `docker/docker-compose.yml` | Prometheus TSDB retention |
| `PROM_UID` | `65534` | `docker/docker-compose.yml` | User id Prometheus runs as, so it can write the TSDB directory; make passes the caller's |
| `PROM_GID` | `65534` | `docker/docker-compose.yml` | Group id Prometheus runs as |
| `AIC_GRAFANA_IMAGE` | `grafana/grafana:13.2.2` | `docker/docker-compose.yml` | Grafana image |
| `AIC_NODE_EXPORTER_IMAGE` | `quay.io/prometheus/node-exporter:v1.12.1` | `docker/docker-compose.yml` | Node exporter image |
| `AIC_AMDGPU_EXPORTER_IMAGE` | `rocm/device-metrics-exporter:v1.5.2` | `docker/docker-compose.yml` | AMD GPU metrics exporter image |
| `AIC_NVME_EXPORTER_IMAGE` | `aic-nvme-exporter:local`; `aic-nvme-exporter:latest` (`.slurm/run-build-distribute.sh`) | `docker/docker-compose.yml` | NVMe exporter image. A local bake tags it `:local`; the Slurm build stores it as `:latest` |
| `AIC_RDMA_EXPORTER_IMAGE` | `aic-rdma-exporter:local`; `aic-rdma-exporter:latest` (`.slurm/run-build-distribute.sh`) | `docker/docker-compose.yml` | RDMA exporter image; tagged like the NVMe one |
| `AIC_HSA_SNOOP_PID_MODE` | `host` (only when unset); `container:${AIC_STACK}-lmcache` (`monitoring/monitoring-lib.sh`) | `docker/docker-compose.yml` | PID namespace of the hsa-snoop container. Empty means private. Only `host` yields kernel-launch and SDMA metrics, because the kprobe reports host PIDs; SPUR refuses that, so make and the monitoring library default it to `container:<stack>-lmcache` there, which exports queue counts only |
| `NODEX_PID_MODE` | (empty) | `docker/docker-compose.yml` | PID namespace of the node exporter; `host` gives per-process CPU statistics where the site permits it |
| `AIC_CPU_SMOKE` | `0` | `monitoring/monitoring-lib.sh` | `1` starts the CPU-only exporter subset (the monitoring smoke test on a node with no GPU) |
| `AIC_MON_SKIP_SERVICES` | (empty) | `monitoring/monitoring-lib.sh` | Space-separated compose services the monitoring library leaves out |
| `COMPOSE_PLUGIN_VERSION` | `v5.5.1` | `monitoring/monitoring-lib.sh` | `docker compose` plugin version installed by `make ensure-compose` and the Slurm jobs when the plugin is missing |
| `PROM_DUMP_OUT` | `${AIC_IMAGE_DIR%/*}/prometheus-dump.md` | `.slurm/run-build-distribute.sh` | Output path of `make prometheus-dump` |
| `PROM_DUMP_WAIT` | `15` | `.slurm/run-build-distribute.sh` | `make prometheus-dump`: seconds between the stack coming up and the scrape |
<!-- END generated:monitoring -->

### Fixed by compose

These are set to a literal in `docker/docker-compose.yml` and are not
user-settable. The readers require them.

| Variable | Value | Read by |
| --- | --- | --- |
| `AIC_L2_DIR`, `AIC_PROM_FILE`, `AIC_PROM_INTERVAL` | see the `l2-monitor` service | `docker/l2-monitor/l2_monitor.py` |
| `LMCACHE_COORDINATOR_PORT`, `COORDINATOR_METRICS_PORT` | `9300`, `9301` | `monitoring/scripts/lmcache_coordinator_entrypoint.py` |
| `LMCACHE_DISABLE_BANNER` | `1` | `monitoring/scripts/lmcache_coordinator_entrypoint.py` |

## Benchmarks

### Client image and `aic-bench`

Every benchmark client runs from one image, `aic-client`
(`docker/client/Dockerfile`, bake target `client`, `make client-build`), and
one command in it, `aic-bench <subcommand>`. `make bench BENCH=<subcommand>
BENCH_ARGS='...'` runs it in the `aic-client` container; `make bench-list`
lists the subcommands. See
[`benchmarks/README.md`](../benchmarks/README.md).

<!-- BEGIN generated:client -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `BENCH` | `list` | `mk/bench.mk` | `aic-bench` subcommand `make bench` runs |
| `BENCH_ARGS` | (empty) | `mk/bench.mk` | Arguments `make bench` passes to the tool behind the subcommand, unchanged |
| `AIC_BENCH_ENDPOINT` | `http://aic-lmcache:8000` | `benchmarks/aic_bench_serve.py` | Server base URL on the `aic` bridge, without `/v1`; `aic-bench --endpoint`. `make bench` passes `BENCH_ENDPOINT`, or the running tiered vLLM |
| `AIC_BENCH_MODEL` | (unset) | `benchmarks/common/kv_client.py` | Served model name; `aic-bench --model`. Empty takes the first model the endpoint lists |
| `AIC_BENCH_OUT` | `/logs/bench` | `benchmarks/aic_bench_serve.py` | Results root in the container; `aic-bench --out`. A run writes to `<out>/<subcommand>/<UTC timestamp>/`, and `/logs` is `logs/` in the checkout |
| `AIC_BENCH_DATASET` | `/tmp/ShareGPT_V3_unfiltered_cleaned_split.json` | `benchmarks/aic_bench_serve.py` | ShareGPT file `aic-bench serve-sim` replays. The `aic-client` container sets it to a path under the Hugging Face cache (`/hf_cache`); it is downloaded when missing |
| `AIC_HSA_SNOOP_URL` | `http://aic-hsa-snoop:9488` | `benchmarks/common/kv_client.py` | hsa-snoop exporter `aic-bench fixed-load` scrapes |
<!-- END generated:client -->

`aic-bench` takes the services next to the endpoint from the endpoint's host.
These name them instead; nothing sets them by default.

| Variable | Default | Read by |
| --- | --- | --- |
| `AIC_BENCH_LMCACHE_HTTP_URL` | `http://<endpoint host>:8080` | `aic-bench reset-test`, `aic-bench agent-kv nvme-sweep` |
| `AIC_BENCH_LMCACHE_METRICS_URL` | `http://<endpoint host>:9091` | `aic-bench agent-simple`, `aic-bench replay-hits` |
| `AIC_BENCH_NIXL_METRICS_URL` | `http://<endpoint host>:19090` | `aic-bench agent-simple` |
| `AIC_BENCH_ROOT` | `/repo/benchmarks` when it holds `aic_bench.py`, else `/opt/aic/benchmarks` | `aic-bench`: directory the benchmark scripts run from |

### Cliff benchmark (`make cliff`)

<!-- BEGIN generated:bench -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `BENCH_ARM` | `kvd_v2` | `Makefile` | Arm `make cliff` runs: `kvd_v2` or `vram_only` |
| `BENCH_ISL` | `20000`; `32` (`.slurm/run-kvbench-cliff.sbatch`) | `Makefile` | Input sequence length, tokens. `make cliff` sets it; the Slurm cliff job leaves the sweep shape to `benchmarks/run_cliff.py` unless it is set |
| `BENCH_SHARED_TOK` | `18000`; `16` (`.slurm/run-kvbench-cliff.sbatch`) | `Makefile` | Shared-prefix length, tokens |
| `BENCH_CONCUR` | `1,2,4,8,16,32,48,64,80,100,128,160,200,250`; `1,2` (`.slurm/run-kvbench-cliff.sbatch`) | `Makefile` | Concurrency ladder. The `cliff-kvd`, `cliff-spur-l2` and `cliff-short` targets each preset a shorter one |
| `BENCH_ITERS` | `3`; `1` (`.slurm/run-kvbench-cliff.sbatch`) | `Makefile` | Iterations per concurrency point |
| `BENCH_ENDPOINT` | (empty) | `Makefile` | vLLM endpoint as seen from the `aic` bridge. Empty resolves to the running tiered vLLM. The AgentX targets read it too |
| `BENCH_MODEL` | `$(VLLM_MODEL)` | `Makefile` | Model name in the benchmark requests |
| `BENCH_PREFIX_MODE` | `per_client` | `.slurm/run-cliff.sbatch` | Prefix sharing in the Slurm cliff job: `per_client` or `shared`. `make cliff-kvd` presets `shared` |
| `BENCH_LOGDIR` | `$(CURDIR)/logs/manual` | `Makefile` | Where non-Slurm runs write results and plots |
<!-- END generated:bench -->

### KVBench

<!-- BEGIN generated:kvbench -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `KVBENCH_MODEL` | `llama-3.1-8b` | `docker/docker-compose.yml` | KVBench model profile |
| `KVBENCH_GPU_PROFILE` | `H100_SXM` | `docker/docker-compose.yml` | KVBench hardware timing profile |
| `KVBENCH_PORT` | `8000` | `docker/docker-compose.yml` | KVBench container port |
| `KVBENCH_CHUNK_SIZE` | `256` | `docker/docker-compose.yml` | KV chunk size, tokens |
| `KVBENCH_LOCAL_CPU_SIZE_GB` | `0.25` | `docker/docker-compose.yml` | Local CPU cache size in the KVBench LMCache config, GB |
| `KVBENCH_READY_S` | `120`; `${KVBENCH_READY_TIMEOUT:-120}` (`.slurm/run-kvbench-cliff.sbatch`) | `Makefile` | Seconds `make cliff-kvbench-local` and `.slurm/run-kvbench-cliff.sbatch` wait for KVBench to answer |
| `AIC_KVBENCH_NODE` | (unset) | `mk/dist.mk` | `make cliff-kvbench-submit`: pin the node |
| `AIC_KVBENCH_CONSTRAINT` | (unset) | `mk/dist.mk` | `make cliff-kvbench-submit`: Slurm `-C` expression (standard Slurm only) |
| `AIC_KVBENCH_TIME` | (unset) | `mk/dist.mk` | `make cliff-kvbench-submit`: job time limit |
<!-- END generated:kvbench -->

### Slurm cliff job (`make cliff-submit`)

`.slurm/run-cliff.sbatch`, submitted by `.slurm/run-build-distribute.sh cliff`.

<!-- BEGIN generated:cliff -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `AIC_CLIFF_ARMS` | `vram,nvme,gds` | `.slurm/run-cliff.sbatch` | Comma list of arms to run: `vram`, `nvme`, `gds`. `make cliff-spur-l2` presets `nvme` |
| `AIC_CLIFF_NODE` | (empty) | `.slurm/run-build-distribute.sh` | Pin the cliff node with `--nodelist` |
| `AIC_CLIFF_GFX` | (empty) | `.slurm/run-build-distribute.sh` | Shorthand for a one-arch `AIC_CLIFF_CONSTRAINT` (`gfx950`) |
| `AIC_CLIFF_CONSTRAINT` | (empty) | `.slurm/run-build-distribute.sh` | Slurm `-C` expression for the cliff node, standard Slurm only. Defaults to `AIC_CLIFF_GFX` in upper case, else `GFX942&NVME`; `none` submits without a constraint |
| `AIC_CLIFF_GPUS` | `1` | `.slurm/run-build-distribute.sh` | GPUs a SPUR cliff submit reserves (`--gpus`). Raise it only with a matching `TENSOR_PARALLEL_SIZE` |
| `AIC_CLIFF_TIME` | `03:00:00` | `.slurm/run-build-distribute.sh` | Cliff job time limit. The make cliff targets each preset their own |
| `AIC_CLIFF_CPUS` | `16` | `.slurm/run-build-distribute.sh` | `--cpus-per-task` of the cliff job |
| `AIC_CLIFF_MEM` | `64G` | `.slurm/run-build-distribute.sh` | `--mem` of the cliff job |
| `AIC_CLIFF_JOB_NAME` | `aic-cliff` | `.slurm/run-build-distribute.sh` | Slurm job name; the make cliff targets each set their own |
| `AIC_MODEL_BIG` | `openai/gpt-oss-120b` | `.slurm/run-cliff.sbatch` | Model the cliff job serves on the large-memory CDNA arches when `VLLM_MODEL` is empty. Must already be in `HF_HOME` |
| `AIC_MODEL_SMALL` | `Qwen/Qwen2.5-3B-Instruct` | `.slurm/run-cliff.sbatch` | Model it serves on every other arch |
| `AIC_NVME_AUTO` | `1` | `.slurm/run-cliff.sbatch` | `1` finds a dedicated local NVMe for the LMCache tiers: a mounted `aic-lmcache` volume, else a raw spare it formats and mounts, else a mounted non-root NVMe, else node-local `/tmp`. `0` forces `/tmp`. Formatting needs passwordless `sudo` |
| `AIC_NVME_MOUNT` | `/mnt/aic-lmcache`; `/mnt/m2m_nobackup` (`Makefile`, SPUR) | `.slurm/run-cliff.sbatch` | Mount point used when a spare NVMe is provisioned; left mounted for reuse |
| `AIC_NVME_LABEL` | `aic-lmcache` | `.slurm/run-cliff.sbatch` | Label of the filesystem that marks a volume as provisioned by the cliff job |
| `AIC_LOCAL_ROOT` | `/tmp/aic-cliff.${SLURM_JOB_ID:-manual}` | `.slurm/run-cliff.sbatch` | Node-local root of the tier directories when `NVME_DATA`, `NFS_DATA` and `GDS_SLAB_DATA` are not set |
| `AIC_LOCAL_CPU` | `false` | `.slurm/run-cliff.sbatch` | `true` puts an LMCache DRAM L1 in front of the NVMe L2 in the `nvme` arm |
| `LMCACHE_MAX_LOCAL_CPU_SIZE` | `40` | `.slurm/run-cliff.sbatch` | Size of that DRAM L1, GB |
| `AIC_LOCAL_DISK_SIZE_GB` | `320` | `.slurm/run-cliff.sbatch` | `local_disk` backend: maximum disk size, GB |
| `AIC_L2_ODIRECT` | `false` | `.slurm/run-cliff.sbatch` | `local_disk` backend: `true` uses `O_DIRECT`, `false` goes through the page cache |
| `AIC_KV_LOAD_FAILURE_POLICY` | (empty) | `.slurm/run-cliff.sbatch` | vLLM `kv_load_failure_policy` for the connector arms: `recompute` or `fail`. Empty leaves vLLM's default |
| `POST_WARMUP_SLEEP_S` | `10` | `.slurm/run-cliff.sbatch` | Seconds to wait after each warm-up wave so asynchronous L2 writes land before the timed iterations |
| `AIC_CLIFF_WARMUP_WAVES` | `2` | `.slurm/run-cliff.sbatch` | Warm-up waves per concurrency level |
| `AIC_CLIFF_RESET_PREFIX_CACHE` | `1` | `.slurm/run-cliff.sbatch` | `1` empties vLLM's GPU prefix cache before every timed iteration of the connector arms |
| `VLM_ROPE_SCALING` | (empty) | `.slurm/run-cliff.sbatch` | Raw JSON for the model's `rope_scaling`, applied to every arm; raise `VLM_MAX_MODEL_LEN` to match |
| `VLM_YARN_FACTOR` | (empty) | `.slurm/run-cliff.sbatch` | Shortcut that builds a `yarn` `rope_scaling` with this factor |
| `VLM_YARN_ORIG_LEN` | `32768` | `.slurm/run-cliff.sbatch` | Native context length that shortcut assumes |
| `AIC_VLLM_READY_TIMEOUT` | `1200` | `.slurm/run-cliff.sbatch` | Seconds the cliff job waits for a vLLM endpoint; raise it for large models on slow storage |
| `LMCACHE_HEARTBEAT_INTERVAL` | `900` | `.slurm/run-cliff.sbatch` | Seconds between the cliff job's LMCache status lines |
| `AIC_VLLM_ARTIFACT` | (empty) | `.slurm/run-cliff.sbatch` | Stored vLLM image artifact a job loads, when it is not the one derived from `AIC_VLLM_IMAGE` and `AIC_IMAGE_DIR` |
| `AIC_LMCACHE_ARTIFACT` | (empty) | `.slurm/run-cliff.sbatch` | Stored LMCache image artifact, likewise |
| `AIC_CLIENT_ARTIFACT` | (empty) | `.slurm/run-cliff.sbatch` | Stored `aic-client` image artifact the cliff job loads for its bench and plot containers, likewise |
<!-- END generated:cliff -->

### AgentX

`make agentx-run`, `agentx-sweep` and `agentx-corpus`; the scripts in
`benchmarks/agentx/` own the defaults and read the same names when run
directly. They also read `AIC_CLIENT_IMAGE` (the image `aiperf` runs from),
`BENCH_ENDPOINT`, `AIC_NETWORK`, `GPU`, `LOG`, `HF_HOME` and `HF_TOKEN`. See
[`benchmarks/agentx/README.md`](../benchmarks/agentx/README.md).

<!-- BEGIN generated:agentx -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `AGENTX_MODEL` | `${VLLM_MODEL:-}` | `benchmarks/agentx/run_agentx.sh` | Served model name in the requests. Empty takes the model the endpoint reports |
| `AGENTX_TOKENIZER` | (empty) | `benchmarks/agentx/run_agentx.sh` | Hugging Face tokenizer `aiperf` counts tokens with. Empty means the model |
| `AGENTX_CONCURRENCY` | `2` | `benchmarks/agentx/run_agentx.sh` | Trajectories replayed at once by `make agentx-run`. Each is one recorded agent session, subagent fan-out included |
| `AGENTX_CONCURRENCIES` | `1 2 4` | `benchmarks/agentx/overnight_sweep.sh` | Space-separated concurrency levels `make agentx-sweep` runs in turn |
| `AGENTX_DURATION` | `1800` | `benchmarks/agentx/run_agentx.sh` | Benchmark duration, seconds. Under 900 the scenario's lock is overridden and the run is not a valid AgentX submission |
| `AGENTX_MAX_CONTEXT` | (empty) | `benchmarks/agentx/run_agentx.sh` | Traces whose peak prompt plus output exceeds this many tokens are dropped. Empty takes the served model's `max_model_len` |
| `AGENTX_DATASET` | `semianalysis_cc_traces_weka_062126` | `benchmarks/agentx/run_agentx.sh` | `aiperf` public dataset alias that is replayed when no local corpus is given |
| `AGENTX_CORPUS` | (empty) | `benchmarks/agentx/run_agentx.sh` | Local `weka-trace` directory to replay instead of `AGENTX_DATASET`. Not a valid AgentX submission |
| `AGENTX_CORPUS_SCALE` | `8` | `benchmarks/agentx/scaled_corpus.sh` | Divisor of the scaled corpus: every token count of the source traces is divided by it. `make agentx-corpus` generates that corpus. `make agentx-run` leaves it empty, which replays `AGENTX_DATASET` as recorded; when set, it replays the scaled corpus, generating it first if needed. A scaled corpus is an AgentX-shaped workload, not the AgentX benchmark |
| `AGENTX_CORPUS_DATASET` | `semianalysisai/cc-traces-weka-062126` | `benchmarks/agentx/scaled_corpus.sh` | Hugging Face dataset repo the scaled corpus is derived from |
| `AGENTX_CORPUS_CACHE` | `${HOME}/.cache/aic-agentx` | `benchmarks/agentx/scaled_corpus.sh` | Directory holding the scaled corpora, one directory per dataset and divisor |
| `AGENTX_SEED` | `42` | `benchmarks/agentx/run_agentx.sh` | `aiperf` random seed: trajectory order and start offsets |
| `AGENTX_SLICE_DURATION` | `30` | `benchmarks/agentx/run_agentx.sh` | Width of the time slices in the `aiperf` export, seconds |
| `AGENTX_SERVER_METRICS` | (empty) | `benchmarks/agentx/run_agentx.sh` | Space-separated Prometheus `/metrics` URLs `aiperf` records besides the endpoint's own. Empty means the stack's LMCache metrics port on the `aic` bridge, or nothing when `BENCH_ENDPOINT` is set; `none` disables it |
| `AGENTX_EXTRA_ARGS` | (empty) | `benchmarks/agentx/run_agentx.sh` | Space-separated arguments appended to `aiperf profile` |
| `AGENTX_ARGS` | (empty) | `mk/bench.mk` | Options `make agentx-run` and `make agentx-sweep` pass to `benchmarks/agentx/run_agentx.sh`, e.g. `--dry-run` |
| `AGENT_SIMPLE_ARGS` | `s1 220 16 480` | `mk/bench.mk` | Arguments `make agent-simple-run` passes to `benchmarks/agent-simple/stress.py`: run id, documents, concurrency, seconds, optional maximum words per document |
| `AGENT_KV_SWEEP_ARGS` | (empty) | `mk/bench.mk` | Extra `agent-kv nvme-sweep` options for `make agent-kv-nvme-sweep`, for example `--delays 0,10,50 --num-books 4` |
<!-- END generated:agentx -->

## Tests

### Smoke test and tiny test (Slurm)

<!-- BEGIN generated:slurm-test -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `AIC_TEST_NODE` | (empty) | `.slurm/run-build-distribute.sh` | Pin the test node with `--nodelist` |
| `AIC_TEST_CONSTRAINT` | (empty) | `.slurm/run-build-distribute.sh` | Slurm `-C` expression for the test node. Derived from `AIC_ROCM_ARCH` when unset, one term per arch joined with `\|`; explicitly empty means none |
| `AIC_TEST_SITE` | (empty) | `.slurm/run-build-distribute.sh` | Site feature put in front of each term of a derived constraint |
| `AIC_TEST_NVME` | (unset) | `.slurm/run-build-distribute.sh` | `1` adds `&NVME` to a derived test constraint |
| `AIC_TEST_EXCLUDE_NODES` | `${AIC_BUILD_EXCLUDE_NODES:-}` | `.slurm/run-build-distribute.sh` | Nodes excluded from the test submits; defaults to `AIC_BUILD_EXCLUDE_NODES` |
| `AIC_TEST_GRES` | `gpu:1` | `.slurm/run-build-distribute.sh` | Standard Slurm GPU request of the smoke and tiny tests |
| `AIC_TEST_GPUS` | `1` | `.slurm/run-build-distribute.sh` | SPUR GPU request of the smoke and tiny tests (`--gpus`) |
| `AIC_TEST_TIME` | `00:45:00` | `.slurm/run-build-distribute.sh` | Smoke-test time limit |
| `AIC_TEST_CPUS` | `8` | `.slurm/run-build-distribute.sh` | `--cpus-per-task` of the smoke test |
| `AIC_TEST_MEM` | `32G` | `.slurm/run-build-distribute.sh` | `--mem` of the smoke test |
| `AIC_SMOKE_EXPORTERS` | `1` | `.slurm/run-build-distribute.sh` | `1` also starts the exporter fleet and Prometheus after the in-image checks and health-checks every `/metrics` endpoint. Informational: it never changes the exit code |
| `AIC_SMOKE_SCRAPE_S` | `45` | `.slurm/run-build-distribute.sh` | Seconds Prometheus scrapes before that health check |
| `AIC_TINY_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | `.slurm/run-build-distribute.sh` | Model `make tiny-test` serves |
| `AIC_TINY_TIME` | `00:40:00` | `.slurm/run-build-distribute.sh` | Tiny-test time limit |
| `AIC_TINY_CPUS` | `8` | `.slurm/run-build-distribute.sh` | `--cpus-per-task` of the tiny test |
| `AIC_TINY_MEM` | `32G` | `.slurm/run-build-distribute.sh` | `--mem` of the tiny test |
| `AIC_TINY_READY_TIMEOUT` | `240` | `.slurm/run-build-distribute.sh` | Readiness attempts, 5 s apart; keep the total under `AIC_TINY_TIME` |
| `AIC_TINY_SCRAPE_S` | `30` | `.slurm/run-build-distribute.sh` | Seconds of metrics scraped after the test |
| `AIC_TINY_EXPORTERS` | `0` | `.slurm/run-build-distribute.sh` | `AIC_EXPORTERS` value for that capture |
<!-- END generated:slurm-test -->

### Reset test (`make vllm-reset-test`)

`benchmarks/vllm_reset_test.py`.

<!-- BEGIN generated:reset-test -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `AIC_TEST_VLLM_URL` | `http://aic-lmcache:8000` | `benchmarks/vllm_reset_test.py` | vLLM base URL, as seen from the `aic` bridge |
| `AIC_TEST_LMCACHE_URL` | `http://aic-lmcache:8080` | `benchmarks/vllm_reset_test.py` | LMCache HTTP API base URL |
| `AIC_TEST_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | `benchmarks/vllm_reset_test.py` | Model name in the requests |
| `AIC_TEST_ANCHORS` | `10` | `benchmarks/vllm_reset_test.py` | Anchor prompts stored first and read back at the end |
| `AIC_TEST_FLOOD` | `50` | `benchmarks/vllm_reset_test.py` | Flood prompts that push the anchors out of L1 |
| `AIC_TEST_ANCHOR_REPS` | `22` | `benchmarks/vllm_reset_test.py` | Repetitions of the preamble sentence per anchor; sets the prompt length |
| `AIC_TEST_MAX_TOKENS` | `20` | `benchmarks/vllm_reset_test.py` | Tokens generated per request |
| `AIC_TEST_TIMEOUT` | `120` | `benchmarks/vllm_reset_test.py` | Per-request timeout, seconds |
| `AIC_TEST_SEED` | `42` | `benchmarks/vllm_reset_test.py` | Random seed of the prompt set |
<!-- END generated:reset-test -->

### Accuracy test (`make accuracy-test`)

The KV-integrity gate: scores gsm8k against a VRAM-only arm and a tiered arm in
one job and asserts that routing KV through DRAM/NVMe did not change the
answers. See [`tests/accuracy/README.md`](../tests/accuracy/README.md) for what
each assertion catches and how to add a model.

The gate scores the full gsm8k split and always runs both arms; the baseline
arm cannot be dropped. `AIC_ACCURACY_LIMIT` caps the items per arm for a
quicker check (`make accuracy-test-very-fast`), at the price of a wider
tolerance.

<!-- BEGIN generated:accuracy -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `AIC_ACCURACY_MODEL` | `${AIC_TINY_MODEL}`; `Qwen/Qwen2.5-0.5B-Instruct` (`tests/accuracy/test_accuracy.py`) | `.slurm/run-build-distribute.sh` | Model to score; must match what the arms serve |
| `AIC_ACCURACY_LIMIT` | (unset) | `tests/accuracy/test_accuracy.py` | Cap on gsm8k items per arm. Unset scores the full split; `make accuracy-test-very-fast` sets `100`, and the floor is widened for a capped run |
| `AIC_ACCURACY_CONCURRENT` | `1` | `tests/accuracy/test_accuracy.py` | `lm_eval` request concurrency. Above 1, answers depend on request timing |
| `AIC_ACCURACY_TIME` | `02:00:00` | `.slurm/run-build-distribute.sh` | Accuracy job time limit |
| `AIC_ACCURACY_CPUS` | `8` | `.slurm/run-build-distribute.sh` | `--cpus-per-task` of the accuracy job |
| `AIC_ACCURACY_MEM` | `32G` | `.slurm/run-build-distribute.sh` | `--mem` of the accuracy job |
| `AIC_ACCURACY_READY_TIMEOUT` | `120` | `.slurm/run-build-distribute.sh` | Endpoint readiness attempts, 5 s apart |
| `AIC_ACCURACY_SCRAPE_S` | `30` | `.slurm/run-build-distribute.sh` | Seconds of metrics scraped after the test |
| `AIC_ACCURACY_EXPORTERS` | `0` | `.slurm/run-build-distribute.sh` | `AIC_EXPORTERS` value for that capture |
| `AIC_ACCURACY_POOL_ROOT` | `/tmp/aic-accuracy.${SLURM_JOB_ID:-manual}` | `.slurm/run-accuracy.sh` | Node-local root of the job's NVMe and NFS pool directories |
| `AIC_ACCURACY_MIN_HIT_PCT` | (empty) | `.slurm/run-accuracy.sh` | Minimum cache hit percentage the tiered arm must report; empty disables the check |
| `AIC_ACCURACY_ALLOW_L1_ONLY` | `0` | `.slurm/run-accuracy.sh` | `1` accepts a run in which nothing was read back from L2 |
| `AIC_ACCURACY_BASELINE_URL` | (unset) | `tests/accuracy/conftest.py` | Baseline arm base URL as seen from the `aic` bridge. Only needed when running pytest by hand; the job supplies it |
| `AIC_ACCURACY_TIERED_URL` | (unset) | `tests/accuracy/conftest.py` | Tiered arm base URL |
| `AIC_ACCURACY_REQUIRED` | (empty) | `tests/accuracy/conftest.py` | `1` turns an unreachable endpoint from a skip into a failure; the CI driver sets it |
| `AIC_ACCURACY_BASELINE_SAMPLES` | (unset) | `tests/accuracy/test_accuracy.py` | Baseline arm's per-question answers (JSON lines), measured earlier in the job; the arms share the GPU, so they are scored one after the other |
| `AIC_ACCURACY_REFERENCE_SAMPLES` | (unset) | `tests/accuracy/test_accuracy.py` | Answers from the baseline arm re-scored with its prefix cache warm; enables the restart assertion |
| `AIC_ACCURACY_BLOCK_SIZE` | `256` | `tests/accuracy/test_accuracy.py` | vLLM `--block-size` the arms ran with. The driver sets it to LMCache's chunk size so both caches reuse the same prefix length |
| `AIC_ACCURACY_SCORE_OUT` | (unset) | `tests/accuracy/test_accuracy.py` | File the measured tiered score is written to, as JSON |
| `AIC_ACCURACY_SAMPLES_OUT` | (unset) | `tests/accuracy/test_accuracy.py` | File the tiered arm's per-question answers are written to, as JSON lines |
<!-- END generated:accuracy -->

### rocjitsu VM test (`make test-rocjitsu-local`)

<!-- BEGIN generated:rocjitsu -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `RJ_QEMU_IMAGE` | `sbates130272/batesste-ci-images-ubuntu-qemu-libvfio-user-sbates-fork:20261024.g9d2be1e-qemu.f91f120-vfu.3719c3a` | `mk/bench.mk` | `qemu` container image. The three images come from one build; bump them as a set |
| `RJ_ROCJITSU_IMAGE` | `sbates130272/batesste-ci-images-ubuntu-rocm-rocjitsu:20261024.g9d2be1e-rocjitsu.98f6a72` | `mk/bench.mk` | rocjitsu emulator container image |
| `RJ_QCOW2_IMAGE` | `sbates130272/batesste-ci-images-ubuntu-qcow2-gen-rocjitsu:20261024.g9d2be1e-vm.resolute-rocjitsu-qm.c079195` | `mk/bench.mk` | Guest disk container image |
| `RJ_ROCJITSU_ARCH` | `gfx1250` | `mk/bench.mk` | Emulated GPU arch |
| `RJ_ROCJITSU_CONFIG` | `gfx1250_mi455x.json` | `mk/bench.mk` | rocjitsu device configuration file |
| `RJ_VM_VCPUS` | `8` | `mk/bench.mk` | Guest CPUs |
| `RJ_VM_MEM_MB` | `16384` | `mk/bench.mk` | Guest memory, MiB |
| `RJ_SSH_PORT` | `12222` | `mk/bench.mk` | Host port forwarded to the guest's SSH |
| `RJ_READY_S` | `300` | `mk/bench.mk` | Seconds to wait for the guest's SSH |
| `RJ_WORK_DIR` | `/tmp/aic-rocjitsu-test` | `mk/bench.mk` | Host working directory |
| `RJ_MODEL` | `HuggingFaceTB/SmolLM2-135M-Instruct` | `mk/bench.mk` | Model served in the guest; it must fit the emulator's 1 GiB of GPU memory |
| `RJ_LMCACHE_L1_SIZE_GB` | `0.03` | `mk/bench.mk` | LMCache L1 size in the guest. Small on purpose, so the cut-down test still overflows it |
| `RJ_NVME_COUNT` | `1` | `mk/bench.mk` | Emulated NVMe devices attached to the guest |
| `RJ_NVME_MOUNT` | `/mnt/lmcache-nvme` | `mk/bench.mk` | NVMe mount point in the guest |
| `RJ_FORCE_BUILD` | `0` | `mk/bench.mk` | `1` rebuilds the images even when arch-matching ones exist |
| `RJ_IMAGE_PREFIX` | (empty) | `mk/bench.mk` | Registry prefix to pull the stack from instead of building it, with its trailing `/` |
| `RJ_IMAGE_ALIAS` | `$(RJ_ROCJITSU_ARCH)` | `mk/bench.mk` | Short alias the images were pushed under |
| `RJ_IMAGE_REF` | `$(word 2,$(_rj_bake))` | `mk/bench.mk` | vLLM image for the emulated arch, as bake tags it |
| `RJ_LMCACHE_IMAGE_REF` | `$(word 3,$(_rj_bake))` | `mk/bench.mk` | LMCache image for the emulated arch |
| `RJ_HSA_SNOOP_IMAGE_REF` | `$(word 6,$(_rj_bake))` | `mk/bench.mk` | hsa-snoop image for the emulated arch |
| `RJ_CLIENT_IMAGE_REF` | `$(word 7,$(_rj_bake))` | `mk/bench.mk` | `aic-client` image loaded into the guest, as bake tags it; built on the host when it is missing |
| `RJ_NCCL_HEARTBEAT_S` | `3600` | `mk/bench.mk` | `NCCL` process-group heartbeat, seconds; long, because an emulated collective is slow |
| `RJ_PROBE` | `0` | `mk/bench.mk` | `1` runs the kernel capability probe before vLLM |
| `RJ_PROBE_ONLY` | `0` | `mk/bench.mk` | `1` stops after the probe |
| `RJ_PROBE_TIMEOUT` | `300` | `mk/bench.mk` | Seconds allowed per probe step |
| `RJ_PROBE_ARGS` | (empty) | `mk/bench.mk` | Extra probe flags |
| `RJ_MAX_MODEL_LEN` | `4096` | `mk/bench.mk` | `--max-model-len` in the guest; keep it within the model's limit |
| `RJ_MAX_BATCHED_TOKENS` | `256` | `mk/bench.mk` | `--max-num-batched-tokens` in the guest; it sets the cost of vLLM's startup profiling pass |
| `RJ_READY_RETRIES` | `600` | `mk/bench.mk` | Readiness polls of the reset test in the guest, 5 s apart |
| `RJ_TEST_ANCHORS` | `2` | `mk/bench.mk` | `AIC_TEST_ANCHORS` of the cut-down reset test |
| `RJ_TEST_FLOOD` | `6` | `mk/bench.mk` | `AIC_TEST_FLOOD` of the cut-down reset test |
| `RJ_TEST_ANCHOR_REPS` | `6` | `mk/bench.mk` | `AIC_TEST_ANCHOR_REPS` of the cut-down reset test; at least 6, so a prompt spans two chunks |
| `RJ_TEST_MAX_TOKENS` | `4` | `mk/bench.mk` | `AIC_TEST_MAX_TOKENS` of the cut-down reset test |
| `RJ_TEST_TIMEOUT` | `1800` | `mk/bench.mk` | `AIC_TEST_TIMEOUT` of the cut-down reset test |
<!-- END generated:rocjitsu -->

### Other Slurm jobs

<!-- BEGIN generated:slurm-other -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `NIGHTLY_TAG` | `nightly` | `.slurm/test-nightly-wheels.sbatch` | `.slurm/test-nightly-wheels.sbatch`: release tag of the wheels under test |
| `WHEEL_DIR` | `/shared_nfs/${USER}/rocm-aic/wheels-nightly` | `.slurm/test-nightly-wheels.sbatch` | `.slurm/test-nightly-wheels.sbatch`: directory holding those wheels |
| `SKIP_ENGINE_INIT` | `0` | `.slurm/test-nightly-wheels.sbatch` | `.slurm/test-nightly-wheels.sbatch`: `1` skips loading the model on the GPU |
| `SKIP_MP_TEST` | `0` | `.slurm/test-nightly-wheels.sbatch` | `.slurm/test-nightly-wheels.sbatch`: `1` skips the LMCache MP-mode test |
| `SKIP_TIER_BENCH` | `0` | `.slurm/test-nightly-wheels.sbatch` | `.slurm/test-nightly-wheels.sbatch`: `1` skips the tier benchmark |
| `ATTN_BACKEND` | `TRITON_ATTN` | `.slurm/test-nightly-wheels.sbatch` | `.slurm/test-nightly-wheels.sbatch`: vLLM attention backend of the wheel test |
| `VLLM_PORT` | `8000` | `.slurm/ext-hit-test.sbatch` | `.slurm/ext-hit-test.sbatch`: vLLM port inside the test's network namespace |
<!-- END generated:slurm-other -->

## Emulation mode and profile capture

See [EMULATE.md](EMULATE.md) for the workflow these belong to.

<!-- BEGIN generated:emulate -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `VLLM_EMULATOR_PROFILE_PACK` | `/opt/llm-emu/profiles/MI300X-Qwen3-8B.json` | `docker/docker-compose.yml` | Profile pack to replay: an in-image path, or `/profiles/<pack>.json` with `EMU_PROFILE_PACK_HOST` |
| `VLLM_EMULATOR_MODE` | `realtime` | `docker/docker-compose.yml` | `realtime` sleeps for the predicted latency; `accelerated` resolves at once |
| `VLLM_EMULATOR_ORACLE_K` | `1` | `docker/docker-compose.yml` | Oracle neighbor count, or `auto` |
| `EMU_PROFILE_PACK_HOST` | `/tmp` | `docker/docker-compose.yml` | Host directory mounted at `/profiles` in the emulator container |
| `EMU_KV_TRANSFER_ARG` | (empty) | `docker/docker-compose.yml` | `--kv-transfer-config` for the emulator. Not `KV_TRANSFER_ARG`, which make exports for the real stack |
| `EMU_IPC_MODE` | `shareable` | `docker/docker-compose.yml` | IPC mode of the emulator. `emulate-mp` needs `container:<stack>-lmcache-emulator` |
| `DEVICE_TYPE` | `cpu` | `docker/docker-compose.yml` | LMCache device type in the `lmcache-emulator` service |
| `VLLM_CPU_MODEL` | `facebook/opt-125m` | `docker/docker-compose.yml` | Model served by the `cpu-smoke-emulator` service |
| `AIC_CONTAINER_USER` | `0:0`; `$(shell id -u):$(shell id -g)` (`Makefile`) | `docker/docker-compose.yml` | `uid:gid` containers run as (default: invoking user). Used by all GPU and emulator services and by the benchmark client (`aic-client`), which is root under plain `docker compose` |
| `RENDER_GID` | `992`; `$(shell stat -c %g /dev/kfd 2>/dev/null)` (`Makefile`) | `docker/docker-compose.yml` | Group of `/dev/kfd`, added as a supplementary group to lmcache and vllm containers so a non-root container can open the GPU. make and the Slurm scripts read it from the node; set it yourself when running `docker compose` directly |
| `AIC_CONTAINER_HOME` | `/tmp` | `docker/docker-compose.yml` | `HOME` inside that container |
| `AIC_CONTAINER_USERNAME` | `root` | `docker/docker-compose.yml` | User name reported inside that container |
| `AIC_EMULATE_MODEL` | `Qwen/Qwen3-8B` | `mk/bench.mk` | Model the emulation tests serve |
| `AIC_EMULATE_READY_S` | `120` | `mk/bench.mk` | `make test-emulate-local`: seconds to wait for the emulator |
| `AIC_EMULATE_ARCH` | `gfx942` | `.slurm/run-build-distribute.sh` | `ROCM_ARCH` of the emulation build; it only selects which PyTorch the base is built for |
| `AIC_EMULATE_PROFILE_PACK` | `/opt/llm-emu/profiles/MI300X-Qwen3-8B.json` | `.slurm/run-build-distribute.sh` | Pack `make emulate-test` replays |
| `AIC_EMULATE_PACK_HOST` | (empty) | `.slurm/run-build-distribute.sh` | Host directory of packs to mount for `make emulate-test` and `make emulate-validate` |
| `AIC_EMULATE_TEST_NODE` | (empty) | `.slurm/run-build-distribute.sh` | Pin the node of the CPU-only emulation jobs |
| `AIC_EMULATE_TEST_CONSTRAINT` | `${AIC_BUILD_CONSTRAINT}` (only when unset) | `.slurm/run-build-distribute.sh` | Slurm `-C` expression of the CPU-only emulation jobs |
| `AIC_EMULATE_TIME` | `00:30:00` | `.slurm/run-build-distribute.sh` | Emulate-test time limit |
| `AIC_EMULATE_CPUS` | `8` | `.slurm/run-build-distribute.sh` | `--cpus-per-task` of the emulate test |
| `AIC_EMULATE_MEM` | `32G` | `.slurm/run-build-distribute.sh` | `--mem` of the emulate test |
| `AIC_EMULATE_READY_TIMEOUT` | `96` | `.slurm/run-build-distribute.sh` | Readiness attempts, 5 s apart |
| `AIC_EMULATE_MP_KV_BYTES` | `8589934592` | `.slurm/run-build-distribute.sh` | `make emulate-mp-test`: emulated KV pool, bytes. It is really allocated, so size it to the node |
| `AIC_EMULATE_MP_L1_GB` | `4` | `.slurm/run-build-distribute.sh` | `make emulate-mp-test`: LMCache L1 size, GiB |
| `AIC_EMULATE_MP_L2_BACKEND` | `nixl_posix` | `.slurm/run-build-distribute.sh` | `make emulate-mp-test`: `AIC_L2_BACKEND` of the emulated stack |
| `AIC_EMULATE_MP_NIXL_POOL` | `512` | `.slurm/run-build-distribute.sh` | `make emulate-mp-test`: POSIX pool slots |
| `AIC_EMULATE_MP_DATA` | `/tmp/aic-emulate-mp` | `.slurm/run-build-distribute.sh` | `make emulate-mp-test`: host directory of the emulated tier volumes |
| `AIC_EMULATE_MP_MONITORING` | `1` | `.slurm/run-build-distribute.sh` | `make emulate-mp-test`: `1` starts the Prometheus sidecar |
| `AIC_EMULATE_MP_BENCH` | (empty) | `.slurm/run-build-distribute.sh` | `make emulate-mp-test`: optional `input_len,output_len,concurrency,num_prompts` sweep run after the cold and warm pair |
| `AIC_EMULATE_MP_BENCH_PREFIX` | (empty) | `.slurm/run-build-distribute.sh` | `make emulate-mp-test`: shared-prefix length for that sweep; empty means every prompt is unique |
| `AIC_EMULATE_MP_TIME` | `00:40:00` | `.slurm/run-build-distribute.sh` | Emulate-mp-test time limit |
| `AIC_EMULATE_MP_CPUS` | `16` | `.slurm/run-build-distribute.sh` | `--cpus-per-task` of the emulate-mp test |
| `AIC_EMULATE_MP_MEM` | `64G` | `.slurm/run-build-distribute.sh` | `--mem` of the emulate-mp test |
| `VLLM_EMUL_MODEL` | `Qwen/Qwen3-8B` | `.github/scripts/run-vllm-emul-perf.sh` | CI emulation performance job: model served |
| `VLLM_EMUL_PROFILE_PACK` | `/opt/llm-emu/profiles/MI300X-Qwen3-8B.json` | `.github/scripts/run-vllm-emul-perf.sh` | CI emulation performance job: pack replayed |
| `VLLM_EMUL_SWEEP` | `1024,128,1,12 1024,128,16,64 4096,128,8,32` | `.github/scripts/run-vllm-emul-perf.sh` | CI emulation performance job: benchmark points |
| `VLLM_EMUL_READY_TIMEOUT` | `96` | `.github/scripts/run-vllm-emul-perf.sh` | CI emulation performance job: readiness attempts, 5 s apart |
| `VLLM_EMUL_ORACLE_K` | `1` | `.github/scripts/run-vllm-emul-perf.sh` | CI emulation performance job: `VLLM_EMULATOR_ORACLE_K` |
| `VLLM_EMUL_OUT_DIR` | (required) | `.github/scripts/run-vllm-emul-perf.sh` | CI emulation performance job: output directory |
<!-- END generated:emulate -->

### Profile capture and validation

<!-- BEGIN generated:capture -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `AIC_CAPTURE_MODEL` | `Qwen/Qwen2.5-3B-Instruct`; `Qwen/Qwen3-8B` (`.slurm/run-build-distribute.sh`) | `mk/bench.mk` | Model served during a profile capture |
| `AIC_CAPTURE_SWEEP` | `128,16,1,12 128,16,8,64 512,64,1,8 512,64,4,32 1024,64,1,8 1024,64,4,32 1024,64,16,64 2048,64,1,6 2048,64,4,24 4096,64,1,4 4096,64,4,16`; `256,64,1,16 256,64,8,64 1024,128,1,12 1024,128,4,32 1024,128,16,64 1024,128,64,128 4096,128,1,8 4096,128,8,32 4096,128,32,64` (`.slurm/run-build-distribute.sh`) | `mk/bench.mk` | `input_len,output_len,concurrency,num_prompts` points driving the capture |
| `AIC_CAPTURE_DIR` | `$(CURDIR)/profiles/captures`; `${AIC_IMAGE_DIR}/profiles` (`.slurm/run-build-distribute.sh`) | `mk/bench.mk` | Where the pack, trace, provenance and benchmark results land |
| `AIC_CAPTURE_HF_HOME` | `${AIC_IMAGE_DIR}/capture-hf` | `.slurm/run-build-distribute.sh` | Hugging Face cache of the Slurm capture job; it holds real weights |
| `AIC_CAPTURE_GPU` | `$(GPU)`; `0` (`.slurm/run-build-distribute.sh`) | `mk/bench.mk` | GPU index used by the capture |
| `IMAGE_REF` | (unset) | `mk/bench.mk` | `make capture-profile-local`: image to capture from; when empty, the `VLLM_IMAGE_REF` bake resolves for this `ROCM_ARCH` |
| `AIC_CAPTURE_GPU_UTIL` | `0.85` | `mk/bench.mk` | `--gpu-memory-utilization` of the capture serve |
| `AIC_CAPTURE_MAX_MODEL_LEN` | `4096`; `8192` (`.slurm/run-build-distribute.sh`) | `mk/bench.mk` | `--max-model-len` of the capture serve; stored in the pack |
| `AIC_CAPTURE_MAX_BATCHED_TOKENS` | `2048`; `4096` (`.slurm/run-build-distribute.sh`) | `mk/bench.mk` | `--max-num-batched-tokens` of the capture serve |
| `AIC_CAPTURE_EXTRA_ARGS` | `--no-enable-prefix-caching` (only when unset) | `.slurm/run-build-distribute.sh` | Extra serve flags for the capture; must match `AIC_VALIDATE_EXTRA_ARGS` |
| `AIC_CAPTURE_WARMUP_SKIP` | `5` | `mk/bench.mk` | `make capture-profile-local`: warm-up steps dropped when the pack is built |
| `AIC_CAPTURE_CONSTRAINT` | `GFX942` (only when unset) | `.slurm/run-build-distribute.sh` | Slurm `-C` expression of the capture job |
| `AIC_CAPTURE_NODE` | (empty) | `.slurm/run-build-distribute.sh` | Pin the capture node |
| `AIC_CAPTURE_TIME` | `01:30:00` | `.slurm/run-build-distribute.sh` | Capture job time limit |
| `AIC_CAPTURE_CPUS` | `32` | `.slurm/run-build-distribute.sh` | `--cpus-per-task` of the capture job |
| `AIC_CAPTURE_MEM` | `128G` | `.slurm/run-build-distribute.sh` | `--mem` of the capture job |
| `AIC_CAPTURE_READY_TIMEOUT` | `180` | `.slurm/run-build-distribute.sh` | Readiness attempts, 5 s apart; the weights download counts |
| `AIC_VALIDATE_PACK` | (empty) | `.slurm/run-build-distribute.sh` | Pack `make emulate-validate` scores; required |
| `AIC_VALIDATE_MODEL` | `${AIC_EMULATE_MODEL}` | `.slurm/run-build-distribute.sh` | Model served for the replay |
| `AIC_VALIDATE_SWEEP` | `1024,128,1,12 1024,128,16,64 4096,128,8,32` | `.slurm/run-build-distribute.sh` | Benchmark points replayed for the comparison against real hardware |
| `AIC_VALIDATE_REAL_DIR` | `${pack_dir}/bench` | `.slurm/run-build-distribute.sh` | Real-hardware benchmark results to compare against; defaults to `bench` next to the pack |
| `AIC_VALIDATE_MAX_MODEL_LEN` | `8192` | `.slurm/run-build-distribute.sh` | `--max-model-len` of the replay |
| `AIC_VALIDATE_MAX_BATCHED_TOKENS` | `4096` | `.slurm/run-build-distribute.sh` | `--max-num-batched-tokens` of the replay |
| `AIC_VALIDATE_EXTRA_ARGS` | `--no-enable-prefix-caching` (only when unset) | `.slurm/run-build-distribute.sh` | Extra serve flags for the replay; must match the capture |
| `AIC_VALIDATE_ORACLE_K` | `1` | `.slurm/run-build-distribute.sh` | `VLLM_EMULATOR_ORACLE_K` of the replay |
<!-- END generated:capture -->

### Read by the emulator itself

These have no default in this repo: compose passes them through to the
llm-emu plugin and the vLLM hooks when they are set.

| Variable | Description |
| --- | --- |
| `VLLM_EMULATOR_ENABLE_ORACLE` | `1` turns on emulation: platform plugin, `torch.cuda` shims, model-load stubs. The emulator services set it |
| `VLLM_EMULATOR_EXECUTOR_HOOK` | `1` swaps `UniProcExecutor.execute_model()` for the profile-pack latency draw. The emulator services set it |
| `VLLM_EMULATOR_ORACLE_MIN_SAMPLES` | Sample floor adaptive-K pools up to. Compose does not forward it, so set it in the emulator container yourself |
| `VLLM_EMULATOR_ORACLE_IGNORE_KV` | `1` forces the 2-axis lookup, ignoring the pack's KV-depth tables |
| `VLLM_EMULATOR_MEMORY` | Emulated device memory, bytes; from the pack when unset. Under `emulate-mp` it is really allocated. Pass it only when set: an empty value is not the same as unset |
| `VLLM_EMULATOR_KV_BUFFERS` | `0` skips the synthesized KV buffers, which leaves the connector inert (llm-emu default `1`) |
| `VLLM_EMULATOR_DEBUG` | `1` logs every oracle lookup |
| `VLLM_EMULATOR_TRACE_STEP_CYCLE` | `1` records one step-latency sample per step on real hardware (profile capture); it does not enable emulation |
| `VLLM_EMULATOR_STEP_TRACE_OUTPUT` | Where that trace is written |

## Miscellaneous

<!-- BEGIN generated:misc -->
| Variable | Default | Owner | Description |
| --- | --- | --- | --- |
| `EXPORT_PREFIX` | `aic-release` | `Makefile` | Name prefix of the `make export` tarball |
| `EXPORT_TARBALL` | `$(CURDIR)/$(EXPORT_PREFIX)-$(_GEN_DATE)-$(_GIT_SHORT_REV)$(_GIT_DIRTY).tar.gz` | `Makefile` | Full path of the `make export` tarball |
<!-- END generated:misc -->
