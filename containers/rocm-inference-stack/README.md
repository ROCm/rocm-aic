<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

## ROCm inference stack container

Image: `rocm/vllm-dev` plus hipFile (sparse `rocm-systems`), LMCache, and NIXL
from [ROCm/nixl][nixl-upstream] cloned over **HTTPS** (default branch at HEAD
when `NIXL_REF` is unset). LMCache may pull CUDA NIXL wheels; those are
removed before the ROCm UCX + meson build installs ROCm/nixl bindings. The
image build **fails** if `import nixl` does not succeed after the NIXL build.

The [llm-d][llm-d-upstream] repo is cloned to `/opt/llm-d/src` for chart paths.
`kubectl` and Helm are installed so you can apply charts from a shell in this
image or copy `/opt/llm-d` to your automation host.

### llm-d and Kubernetes

Use `LLM_D_GIT_URL`, `LLM_D_REF`, and optional `LLM_D_SHA` to pin the checkout.
Typical flow: point `KUBECONFIG` at your cluster, `cd /opt/llm-d/src`, then run
`helm upgrade --install` against the chart paths documented upstream for your
llm-d release. The vLLM workload remains the inference process; the full llm-d
control plane runs on Kubernetes, not inside the vLLM container.

### Private or forked NIXL

Use an HTTPS `NIXL_GIT_URL` reachable during `docker build` (CI token, mirror,
or public fork). This Dockerfile does **not** use BuildKit SSH mounts; `git@`
URLs are rejected in `scripts/clone-nixl.sh`. Ansible rejects `git@` for
`inference_nixl_git_url` to match.

### Local build

```bash
export DOCKER_BUILDKIT=1
docker buildx build -f Dockerfile -t rocm-inference-stack:latest .
```

Ansible playbook [`playbooks/inference-image.yml`][inf-playbook] (tags
`inference-container`, `inference-image-build`, `inference-image-deploy`) runs
the same `docker buildx build` with `--build-arg` values from role defaults.

## Build arguments

| ARG | Default | Notes |
|-----|---------|-------|
| `HIPFILE_SHA` | `b2509f2` | `rocm-systems` sparse checkout |
| `LMCACHE_GIT_URL` | amd LMCache fork | |
| `LMCACHE_SHA` | `bef2e13` | |
| `GPU_ARCHS` | `gfx942` | Semicolon list for builders without GPUs |
| `NIXL_GIT_URL` | `https://github.com/ROCm/nixl.git` | |
| `NIXL_REF` | *(empty)* | Branch, tag, or commit; empty = default branch |
| `LLM_D_GIT_URL` / `LLM_D_REF` | llm-d `main` | |
| `LLM_D_SHA` | *(empty)* | Optional commit after shallow clone |

<!-- References -->

[nixl-upstream]: https://github.com/ROCm/nixl
[llm-d-upstream]: https://github.com/llm-d/llm-d
[inf-playbook]: ../../ansible/playbooks/inference-image.yml
