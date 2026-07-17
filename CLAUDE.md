# CLAUDE.md — AMD Infinity Context (AIC)

## SPUR Cluster Access

SPUR commands (`srun`, `sinfo`, `squeue`) require the controller address
to be exported. This is **not** set automatically in non-login shells
(e.g. Claude Code):

```bash
export SPUR_CONTROLLER_ADDR=http://crs-m2m-cpu-spur-005.crusoe.amd.com:6817
```

Add this before any `srun` invocation, or it will fail with
"Connection refused on localhost:6817".

- **Partition:** `amd-spur`
- **No `--account` flag needed** for `srun`
- Node naming convention: `crsuse2-m2m-NNN`

## ROCm 7.14 torch wheel integration — known issues (2026-07-20)

The Dockerfile was updated to use AMD's native ROCm 7.14 wheel index
(`repo.amd.com/rocm/whl-multi-arch/`) for PyTorch. Testing on the SPUR
cluster (MI355X / gfx950) revealed a **blocking incompatibility**:

**`torch 2.12.0+rocm7.14.0` ships Triton 3.7.1, which breaks vLLM 0.25.1.**
vLLM's engine core fails to start on gfx950 nodes — Triton kernel calls
fail because Triton is disabled at startup due to circular import side-effects
in vLLM's platform detection code.

### Quick unblock options

1. **Use gfx942 (MI300X) nodes** — the Triton issue manifests on gfx950
   (MI355X) specifically; gfx942 nodes have not been fully tested but the
   hipErrorInvalidImage issue is resolved with `AIC_ROCM_ARCH=gfx942`.

2. **Revert torch to pytorch.org** — change the Dockerfile torch install back
   to `torch 2.13.0+rocm7.2` from `download.pytorch.org/whl/rocm7.2`.
   That wheel ships a Triton version compatible with vLLM 0.25.1.

3. **Upgrade vLLM** — newer vLLM versions (0.28+) may support Triton 3.7.

### SPUR cluster cliff submission quirks

- Pass `--gres=` (empty) to override the embedded `#SBATCH --gres=gpu:1` —
  SPUR's scheduler has no GPU GRES configured, so the directive causes
  immediate job cancellation.
- Do NOT use `--no-requeue` — SPUR sbatch doesn't support it.
- Submit with `--chdir=/shared_nfs/$USER` when `/home` is at quota.
- Use `SLURM_SUBMIT_DIR=/path/to/repo` to tell the cliff script where the repo is.
- Pin to a node with no pre-existing containers: `--nodelist=crsuse2-m2m-042`
  (as of 2026-07-20 crsuse2-m2m-036 has a long-running primus container
  that blocks GPU access for new containers).
- All SPUR nodes are **MI355X (gfx950)**, not MI300X (gfx942).

### Full analysis

See `.docs-remove/rocm-7.14-torch-wheel-integration.md` for the complete
11-bug chain discovered during SPUR testing, including root causes and fix
status for each.
