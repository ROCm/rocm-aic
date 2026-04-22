<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

## ROCm inference stack container

Image: `rocm/vllm-dev` plus hipFile (sparse `rocm-systems`), LMCache, NIXL from
`git@github.com:ROCm/nixl` (private repos need **your** GitHub SSH key),
ROCm/UCX, `kubectl`, and Helm. [llm-d][llm-d] sources live under `/opt/llm-d/src`
for chart references.

## Private NIXL (your SSH key)

If `ROCm/nixl` is only visible with credentials you load into `ssh-agent`, the
host that runs `docker build` / Ansible must use **BuildKit** and forward the
agent (`--ssh=default`). Keys are never copied into the image or the repo.

```bash
export DOCKER_BUILDKIT=1
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
ssh -T git@github.com
docker buildx build --ssh=default -f Dockerfile -t rocm-inference-stack:latest .
```

[`playbooks/inference-image.yml`](../../ansible/playbooks/inference-image.yml)
sets `DOCKER_BUILDKIT=1` and adds `--ssh=default` when `inference_nixl_git_url`
matches `git@`. Run `ansible-playbook` from the same shell where `ssh-add` was
run so `SSH_AUTH_SOCK` is set.

To avoid SSH, set `inference_nixl_git_url` to an HTTPS URL and extend the
Dockerfile with a [BuildKit secret][bk-secret] for a token (not wired in this
role yet).

## Build arguments

| ARG | Default | Notes |
|-----|---------|--------|
| `HIPFILE_SHA` | `b2509f2` | `rocm-systems` sparse checkout |
| `LMCACHE_GIT_URL` | amd LMCache fork | |
| `LMCACHE_SHA` | `bef2e13` | |
| `GPU_ARCHS` | `gfx942` | Semicolon list for builders without GPUs |
| `NIXL_GIT_URL` | `git@github.com:ROCm/nixl.git` | |
| `NIXL_REF` | *(empty)* | Shallow clone of default branch |
| `LLM_D_GIT_URL` / `LLM_D_REF` | llm-d `main` | |

## Ansible

From `ansible/`: role `inference_container`; playbook
`playbooks/inference-image.yml` with tags `inference-image-build` and
`inference-image-deploy`. Deploy installs `docker.io`, copies the tarball, and
runs `docker load`.

<!-- References -->

[bk-secret]: https://docs.docker.com/build/building/secrets/
[llm-d]: https://github.com/llm-d/llm-d
