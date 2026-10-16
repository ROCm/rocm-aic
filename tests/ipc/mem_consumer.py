#!/usr/bin/env python3
"""Cross-container HIP VMM memory-IPC — consumer. Receive the dmabuf fd via
SCM_RIGHTS, import it, read the GPU memory back, and verify the 0xAB pattern."""
import os, sys, ctypes, socket, time
sys.path.insert(0, "/vmem")
import rocm_vmem_ipc as R   # noqa: E402

SIZE, D2H = 1 << 20, 2
LEG = os.environ.get("HSA_ENABLE_IPC_MODE_LEGACY", "<unset>")
hip = R._load_hip(); hip.hipSetDevice(0)
print(f"[mem-cons] LEGACY={LEG}", flush=True)

sp = "/ipc/mem.sock"
for _ in range(120):
    if os.path.exists("/ipc/prod.ready") and os.path.exists(sp):
        break
    time.sleep(1)
c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
c.connect(sp)
msg, fds, _, _ = socket.recv_fds(c, 64, 1)
asz = int(msg.decode()); imp_fd = fds[0]
print(f"[mem-cons] received fd={imp_fd} size={asz} via SCM_RIGHTS", flush=True)

try:
    imp_h, imp_va = R._vmem_import(hip, imp_fd, asz, 0)
    print(f"[mem-cons] hipMemImportFromShareableHandle OK (va=0x{imp_va:x})", flush=True)
except Exception as e:                           # noqa: BLE001
    print(f"[mem-cons] RESULT [BROKEN] import err={type(e).__name__}: {e}", flush=True)
    sys.exit(1)

out = (ctypes.c_ubyte * SIZE)()
rc = hip.hipMemcpy(out, ctypes.c_void_p(imp_va), ctypes.c_size_t(SIZE), D2H)
hip.hipDeviceSynchronize()
bad = sum(1 for b in out if b != 0xAB)
verdict = "WORKS" if (rc == 0 and bad == 0) else "CORRUPT"
print(f"[mem-cons] RESULT [{verdict}] d2h_rc={rc} mismatched={bad}/{SIZE}", flush=True)
sys.exit(0 if verdict == "WORKS" else 3)
