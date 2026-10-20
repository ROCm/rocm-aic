#!/usr/bin/env python3
"""Same-process HIP VMM memory-IPC round-trip (the dmabuf path foundation).

alloc VMM -> write 0xAB -> hipMemExportToShareableHandle(fd) ->
hipMemImportFromShareableHandle(fd) -> read back -> verify. This is exactly the
export/import pair the LMCache memory path relies on. Run under both
HSA_ENABLE_IPC_MODE_LEGACY settings to confirm whether LEGACY breaks it
(the claim: LEGACY replaces the dmabuf path at runtime).

Reuses the VMM helpers from docker/lmcache/rocm_vmem_ipc.py (mounted at /vmem).
"""
import os, sys, ctypes

sys.path.insert(0, "/vmem")
import rocm_vmem_ipc as R   # noqa: E402

LEG = os.environ.get("HSA_ENABLE_IPC_MODE_LEGACY", "<unset>")
SIZE = 1 << 20
H2D, D2H = 1, 2

hip = R._load_hip()
hip.hipSetDevice(0)
print(f"[mem-selftest] LEGACY={LEG}", flush=True)

def die(stage, e):
    print(f"[mem-selftest] RESULT [BROKEN] stage={stage} err={type(e).__name__}: {e}",
          flush=True)
    sys.exit(1)

try:
    h, va, asz = R._vmem_alloc(hip, SIZE, 0)
    print(f"[mem] vmem_alloc OK (va=0x{va:x} size={asz})", flush=True)
except Exception as e:                           # noqa: BLE001
    die("alloc", e)

src = (ctypes.c_ubyte * SIZE)(*([0xAB] * SIZE))
hip.hipMemcpy(ctypes.c_void_p(va), src, ctypes.c_size_t(SIZE), H2D)
hip.hipDeviceSynchronize()

try:
    fd = R._vmem_export(hip, h)
    print(f"[mem] hipMemExportToShareableHandle OK (fd={fd})", flush=True)
except Exception as e:                           # noqa: BLE001
    die("export", e)

try:
    imp_h, imp_va = R._vmem_import(hip, fd, asz, 0)
    print(f"[mem] hipMemImportFromShareableHandle OK (va=0x{imp_va:x})", flush=True)
except Exception as e:                           # noqa: BLE001
    die("import", e)

out = (ctypes.c_ubyte * SIZE)()
rc = hip.hipMemcpy(out, ctypes.c_void_p(imp_va), ctypes.c_size_t(SIZE), D2H)
hip.hipDeviceSynchronize()
bad = sum(1 for b in out if b != 0xAB)
verdict = "WORKS" if (rc == 0 and bad == 0) else "CORRUPT"
print(f"[mem-selftest] RESULT [{verdict}] d2h_rc={rc} mismatched={bad}/{SIZE}", flush=True)
