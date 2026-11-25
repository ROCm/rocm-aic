# AGENTS.md

Guidance for AI coding agents and human contributors working in this repository.
For what the project is and how to run it, see [README.md](README.md) and
[docs/QUICK_START.md](docs/QUICK_START.md); for how the containers fit together,
see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Common commands

Run from the repository root. `make help` lists every target.

```bash
make build ROCM_ARCH=gfx950
make up HF_TOKEN=hf_... NVME_DATA=/path/to/nvme VLLM_MODEL=<model>
make down
make venv                            # host-side Python environment (.venv)
make list-vars                       # values make would use on this host
```

Unit tests:

```bash
python3 -m unittest -v tests.test_aic_bake_env tests.test_gen_env_reference tests.test_runtime_defaults
```

## Component versions

- **Pins live in `docker/docker-bake.hcl`.** Change a version there, not in a
  Dockerfile. Any `ARG` the bake file supplies has no default in the Dockerfile.
- **The README is generated from the pins.** Its badges and stack table are
  rewritten from the bake file by the `readme-versions` pre-commit hook.
- **Patches are numbered and minimal.** Prefer sending fixes upstream; keep
  local patches small and named for what they fix.

## Conventions

- **Install the pre-commit hooks** once with `.venv/bin/pre-commit install`
  (after `make venv`). They run `shellcheck`, `hadolint`, and regenerate the
  README versions and [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).
- **After adding or changing a build or runtime variable**, run `make env-docs`.
- **Spelling is checked** on Markdown. Add legitimate new terms to `.wordlist.txt`.
- **Commits** follow Conventional Commits: `type(scope): subject`, imperative,
  50 characters or fewer.
- **Do not hard-code cluster details.** Slurm/SPUR usage is described in
  [docs/SLURM_SPUR.md](docs/SLURM_SPUR.md); host names, partitions and paths
  come from variables, not the repository.
