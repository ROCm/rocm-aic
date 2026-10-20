#!/usr/bin/env python3
"""Cross-container HIP IPC event micro-benchmark — consumer side.

Reads the producer's event IPC handle from the shared /ipc volume and imports
it via torch.cuda.Event.from_ipc_handle() 40 times to probe:
  * whether cross-container event import works at all under this mode, and
  * whether it survives >32 reuses (the IPCEventEmulated 32-slot signal ring,
    patched by patches/hip-ipc-event-ring-fix.patch).

Exit 0 if any import succeeded, non-zero otherwise. Prints a one-line RESULT.
"""
import os, sys, time, torch

IPC = "/ipc"
N = 40
LEG = os.environ.get("HSA_ENABLE_IPC_MODE_LEGACY", "<unset>")
print(f"[consumer] torch={torch.__version__} hip={torch.version.hip} "
      f"LEGACY={LEG}", flush=True)

torch.cuda.init()
dev = torch.device("cuda", torch.cuda.current_device())

for _ in range(120):
    if os.path.exists(f"{IPC}/producer.ready"):
        break
    if os.path.exists(f"{IPC}/producer.error"):
        print("[consumer] producer failed to export handle; aborting", flush=True)
        sys.exit(2)
    time.sleep(1)
else:
    print("[consumer] timed out waiting for producer", flush=True)
    sys.exit(2)

with open(f"{IPC}/evt.handle", "rb") as f:
    handle = f.read()
print(f"[consumer] read handle ({len(handle)} bytes); importing {N}x", flush=True)

ok = fail = 0
first_fail_at = None
first_err = None
for i in range(N):
    try:
        e = torch.cuda.Event.from_ipc_handle(dev, handle)
        e.synchronize()
        ok += 1
    except Exception as ex:                     # noqa: BLE001
        fail += 1
        if first_fail_at is None:
            first_fail_at = i + 1
            first_err = f"{type(ex).__name__}: {ex}"

verdict = "WORKS" if ok == N else ("PARTIAL" if ok else "BROKEN")
print(f"[consumer] RESULT [{verdict}] ok={ok}/{N} fail={fail} "
      f"first_fail_at={first_fail_at} err={first_err}", flush=True)
with open(f"{IPC}/consumer.done", "w") as f:
    f.write(f"{verdict} ok={ok} fail={fail}")
sys.exit(0 if ok else 3)
