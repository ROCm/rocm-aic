# HIP IPC Capability Report — gfx1201 (RX 9070 XT / Navi 48)
Date: 2026-10-09  
Host: Ubuntu 24.04, ROCm 7.14.1 (container) / 7.16 (host), amdgpu-dkms 7.1.3  
GPU: AMD Radeon RX 9070 XT (gfx1201, 16 GiB VRAM, large BAR enabled)  
Platform: Threadripper PRO 7955WX / TRX50

---

## Executive Summary

Cross-container zero-copy GPU IPC **works on gfx1201** using the HIP Virtual Memory
Management (VMM) API with `hipMemCreate` / `hipMemExportToShareableHandle` /
`hipMemImportFromShareableHandle` + Unix socket SCM_RIGHTS fd passing.

The standard `hipIpcOpenMemHandle` path does NOT work because ROCR's internal
IPC socket is namespace-scoped and cannot cross container boundaries. The HIP VMM
path bypasses ROCR's socket entirely by giving the caller a real dmabuf file
descriptor that can be transferred via SCM_RIGHTS.

---

## Why `hipIpcOpenMemHandle` Fails Cross-Container

1. `hipIpcGetMemHandle` (exporter) calls `AMDKFD_IOC_EXPORT_DMABUF` → gets a dmabuf
   fd → stores the kernel dmabuf object ID in the opaque handle → **closes the fd**
2. ROCR starts an internal Unix socket server keyed to the exporting process
3. `hipIpcOpenMemHandle` (importer) connects to ROCR's socket server → receives the
   dmabuf fd via SCM_RIGHTS → calls `AMDKFD_IOC_IMPORT_DMABUF` → maps the buffer
4. **Cross-container failure**: ROCR's socket server uses a process-namespace-scoped
   path. Even with `--pid=host`, the two containers open `/dev/kfd` independently,
   creating separate KFD process contexts. The socket server is not reachable
   cross-container. `AMDKFD_IOC_IMPORT_DMABUF` is **never called** — ROCR aborts
   before the syscall.

`HSA_ENABLE_IPC_MODE_LEGACY=1` (baked into our base image) additionally prevents
`hipIpcGetMemHandle` from working on gfx1201 entirely. This must be removed from
the base image ENV.

PCIe topology (`pci_p2pdma_distance = 0`) and GPU pool sizes are **not the issue**.

---

## Working Solution: HIP VMM + SCM_RIGHTS

**Sender (vLLM process):**
```python
# Allocate with POSIX handle type
prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor  # 0x1
hip.hipMemCreate(&handle, size, &prop, 0)
hip.hipMemAddressReserve(&va, size, 0, None, 0)
hip.hipMemMap(va, size, 0, handle, 0)
hip.hipMemSetAccess(va, size, &access_desc, 1)
# Export to real dmabuf fd
hip.hipMemExportToShareableHandle(&fd, handle, 0x1, 0)
# Send fd via Unix socket SCM_RIGHTS
sendmsg(sock, SCM_RIGHTS=[fd])
```

**Receiver (lmcache process):**
```python
# Receive fd via SCM_RIGHTS
dmabuf_fd = recvmsg(sock, SCM_RIGHTS)
# Import
hip.hipMemImportFromShareableHandle(&hip_handle, dmabuf_fd, 0x1)
hip.hipMemAddressReserve(&va, size, 0, None, 0)
hip.hipMemMap(va, size, 0, hip_handle, 0)
hip.hipMemSetAccess(va, size, &access_desc, 1)
# Now va points to the sender's GPU memory — zero-copy
```

**Confirmed results (cross-container, gfx1201):**
- Pattern `[0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]` written by sender
- Read back correctly by receiver: `✓ MATCH — zero-copy cross-container GPU IPC WORKS`
- `hipMemImportFromShareableHandle rc=0 (hipSuccess)`
- `hipMemMap rc=0 (hipSuccess)`
- `hipMemcpy D2H rc=0`

Container flags required: `--device=/dev/kfd --device=/dev/dri --ipc=host`  
(`--pid=host` not required for VMM path; `--network=host` not required)  
Env: `HSA_ENABLE_IPC_MODE_LEGACY=""` (unset, or remove from image)

---

## Results Matrix

| Test | Result | Notes |
|------|--------|-------|
| `hipIpcGetMemHandle` (no legacy flag) | ✓ | Fails with `HSA_ENABLE_IPC_MODE_LEGACY=1` |
| `hipIpcOpenMemHandle` cross-container | ✗ | ROCR socket namespace-scoped |
| `hsa_amd_ipc_memory_create/attach` cross-container | ✗ | Same ROCR socket issue |
| `hsa_amd_ipc_memory_create/attach` same container | ✓ | Works |
| `hipMemCreate` + `hipMemExportToShareableHandle` | ✓ | Real dmabuf fd |
| `hipMemImportFromShareableHandle` cross-container | ✓ | **WORKS via SCM_RIGHTS** |
| `hipIpcGetEventHandle` | ✗ | `hipErrorInvalidConfiguration` unconditional |
| `pci_p2pdma_distance_many(gpu, gpu)` | ✓ = 0 | On-die switch, P2P fine |

---

## Implementation Plan for LMCache MP Connector

### Does vLLM need patching?

**No.** All changes are in lmcache files that run inside the vLLM EngineCore process
(`ipc_wrapper.py`, `worker_transfer.py`) plus the lmcache server
(`lmcache_driven_transfer.py`). vLLM itself is unchanged.

### Architecture

The KV cache tensors in vLLM are `hipMalloc`-backed (`torch.zeros(..., device='cuda')`).
The VMM path requires `hipMemCreate`-allocated memory (different allocator).

**Solution**: The ROCm IPC wrapper allocates a `hipMemCreate` shadow buffer per KV
tensor, copies the KV data into it, exports the shadow buffer via SCM_RIGHTS.
lmcache server maps the shadow buffer and reads KV data directly (zero-copy from
lmcache's perspective once the initial copy from hipMalloc→VMEM is done).

Alternatively (lower latency): vLLM allocates KV cache via `hipMemCreate` from the
start. This requires a vLLM patch to change the KV allocator, but eliminates the
shadow copy. Implement the shadow approach first; switch to native VMEM later.

### Files to patch (lmcache)

1. **`lmcache/v1/platform/cuda/ipc_wrapper.py`** — add `RocmVmemIPCWrapper`
2. **`lmcache/v1/platform/kv_wrap.py`** — use `RocmVmemIPCWrapper` on ROCm
3. **`lmcache/v1/multiprocess/transfer_context/worker_transfer.py`** — send fds via
   sidecar socket after registration
4. **`lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py`** — receive fds,
   map VMEM buffers for store/retrieve

### Sidecar socket

A Unix domain socket at `/tmp/aic-ipc-fds.sock` (bind-mounted into both containers)
carries the dmabuf fds via SCM_RIGHTS alongside the main ZMQ registration channel.

---

## Immediate Actions

1. **Remove `HSA_ENABLE_IPC_MODE_LEGACY=1` from `docker/base/Dockerfile`** —
   add as compose `environment:` override for CDNA services only.
2. **Implement `RocmVmemIPCWrapper`** in lmcache (see patch in
   `patches/lmcache/17-lmcache-rocm-vmem-ipc.patch`).
3. **Add `/tmp/aic-ipc` shared volume** to docker-compose for the sidecar socket.
