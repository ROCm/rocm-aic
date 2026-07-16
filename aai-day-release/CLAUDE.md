# CLAUDE.md — aai-day-release

## SPUR Cluster Access

SPUR commands (`srun`, `sinfo`, `squeue`) require the controller address to be exported.
This is **not** set automatically in non-login shells (e.g. Claude Code):

```bash
export SPUR_CONTROLLER_ADDR=http://crs-m2m-cpu-spur-005.crusoe.amd.com:6817
```

Add this before any `srun` invocation, or it will fail with "Connection refused on localhost:6817".

- **Partition:** `amd-spur`
- **No `--account` flag needed** for `srun`
- Node naming convention: `crsuse2-m2m-NNN`
- Full cluster hardware inventory: `docs/amd-spur-cluster.md`
