#!/usr/bin/env python3
"""Cross-container HIP VMM memory-IPC — producer. Alloc VMM, fill 0xAB, export a
dmabuf fd, and hand it to the consumer container via SCM_RIGHTS over a Unix
socket on the shared /ipc volume."""
import os, sys, ctypes, socket, time
sys.path.insert(0, "/vmem")
import rocm_vmem_ipc as R   # noqa: E402

SIZE, H2D = 1 << 20, 1
LEG = os.environ.get("HSA_ENABLE_IPC_MODE_LEGACY", "<unset>")
hip = R._load_hip(); hip.hipSetDevice(0)
print(f"[mem-prod] LEGACY={LEG}", flush=True)

h, va, asz = R._vmem_alloc(hip, SIZE, 0)
src = (ctypes.c_ubyte * SIZE)(*([0xAB] * SIZE))
hip.hipMemcpy(ctypes.c_void_p(va), src, ctypes.c_size_t(SIZE), H2D)
hip.hipDeviceSynchronize()
fd = R._vmem_export(hip, h)
print(f"[mem-prod] exported dmabuf fd={fd} size={asz}", flush=True)

sp = "/ipc/mem.sock"
try: os.unlink(sp)
except FileNotFoundError: pass
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(sp); srv.listen(1)
with open("/ipc/prod.ready", "w") as f: f.write("1")
print("[mem-prod] waiting for consumer to connect...", flush=True)
conn, _ = srv.accept()
socket.send_fds(conn, [str(asz).encode()], [fd])
print("[mem-prod] fd sent via SCM_RIGHTS; holding memory", flush=True)
time.sleep(8)
print("[mem-prod] done", flush=True)
