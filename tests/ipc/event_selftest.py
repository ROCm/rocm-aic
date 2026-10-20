#!/usr/bin/env python3
"""Same-process HIP IPC event baseline.

Export + import an interprocess event handle within ONE process. If this fails,
cross-container is hopeless; if it works, the cross-container failure is a
namespace/context issue, not a basic-API issue. Run under both LEGACY settings.
"""
import os, torch

LEG = os.environ.get("HSA_ENABLE_IPC_MODE_LEGACY", "<unset>")
print(f"[selftest] torch={torch.__version__} hip={torch.version.hip} LEGACY={LEG}",
      flush=True)
torch.cuda.init()
dev = torch.device("cuda", torch.cuda.current_device())
s = torch.cuda.Stream(device=dev)
with torch.cuda.stream(s):
    _ = torch.ones(4096, device=dev)
    evt = torch.cuda.Event(interprocess=True)
    evt.record(s)

try:
    h = evt.ipc_handle()
    print(f"[selftest] ipc_handle() OK ({len(h)} bytes)", flush=True)
except Exception as e:                           # noqa: BLE001
    print(f"[selftest] ipc_handle() FAILED: {type(e).__name__}: {e}", flush=True)
    raise SystemExit(1)

ok = fail = 0
err = None
for _ in range(40):
    try:
        e2 = torch.cuda.Event.from_ipc_handle(dev, h)
        e2.synchronize()
        ok += 1
    except Exception as ex:                      # noqa: BLE001
        fail += 1
        err = err or f"{type(ex).__name__}: {ex}"
verdict = "WORKS" if ok == 40 else ("PARTIAL" if ok else "BROKEN")
print(f"[selftest] same-process import RESULT [{verdict}] ok={ok}/40 fail={fail} err={err}",
      flush=True)
