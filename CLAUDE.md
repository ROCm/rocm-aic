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

## Historical ROCm 7.14 torch wheel integration issue (2026-07-20)

The Dockerfile was temporarily updated to use AMD's native ROCm 7.14 wheel index
(`repo.amd.com/rocm/whl-multi-arch/`) for PyTorch. Testing on the SPUR
cluster (MI355X / gfx950) revealed a **blocking incompatibility**:

**`torch 2.12.0+rocm7.14.0` ships Triton 3.7.1, which breaks older vLLM 0.25.1.**
vLLM's engine core fails to start on gfx950 nodes — Triton kernel calls
failed because Triton was disabled at startup due to circular import side-effects
in vLLM's platform detection code.

The current Dockerfile intentionally uses `torch 2.13.0+rocm7.2` with vLLM
`v0.27.1`; validate changes to either pin on an MI355X before changing this
historical workaround.

vLLM v0.27.1 no longer calls `logger.warning_once()` from
`vllm.platforms.rocm._get_gcn_arch()` when the amdsmi probe fails, so the old
ROCm-AIC circular-import patch was retired. The SPUR validation must confirm
that the vLLM ROCm-platform import and engine startup remain healthy.

### Historical unblock options

1. **Use gfx942 (MI300X) nodes** — the Triton issue manifests on gfx950
   (MI355X) specifically; gfx942 nodes have not been fully tested but the
   hipErrorInvalidImage issue is resolved with `AIC_ROCM_ARCH=gfx942`.

2. **Use the current pytorch.org pin** — the Dockerfile already installs
   `torch 2.13.0+rocm7.2` from `download.pytorch.org/whl/rocm7.2`; retain it
   unless SPUR validation proves a different compatible pair is needed.

3. **Upgrade vLLM further** — only after validating the current `v0.27.1`
   stack; newer vLLM versions may support the ROCm 7.14 Triton combination.

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

## gfx1201 (RDNA4 / RX 9070 XT) local development notes (2026-09-03)

The RX 9070 XT (Navi 48, gfx1201, 16 GB) is supported in the default multi-arch
build (`AIC_ROCM_ARCH` default includes `gfx1201`). Key differences from CDNA:

- **Triton**: `torch 2.13.0+rocm7.2` from pytorch.org works on gfx1201 without
  the Triton 3.7.1 incompatibility that affects gfx950 on older KFD. The current
  Dockerfile pin is correct; do not switch back to `repo.amd.com` for gfx1201.
- **NIXL AIS_MT**: hipFile P2PDMA is CDNA-only; use `AIC_L2_BACKEND=nixl_posix`
  on gfx1201 nodes (the SPUR default already does this).
- **VFIO**: the GPU is typically VFIO-passed to a VM on workstations. Rebind to
  `amdgpu` before running Docker containers:

  ```bash
  echo "0000:10:00.0" | sudo tee /sys/bus/pci/devices/0000:10:00.0/driver/unbind
  echo "0000:10:00.0" | sudo tee /sys/bus/pci/drivers/amdgpu/bind
  ```

- **Profile capture** on a local gfx1201 node (no cluster required):

  ```bash
  make capture-profile-local \
    IMAGE_REF=<image> ROCM_ARCH=gfx1201 \
    AIC_CAPTURE_MODEL=Qwen/Qwen2.5-3B-Instruct \
    HF_TOKEN_FILE=~/.cache/huggingface/token
  ```
  Output lands in `profiles/captures/`. The captured gfx1201 pack
  (`Qwen-Qwen2.5-3B-Instruct-20260902-182346.json`) is the first RDNA4 pack
  in the repo; see `prometheus-dump.md` for emulated-vs-real latency numbers.
