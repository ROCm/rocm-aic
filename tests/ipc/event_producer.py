#!/usr/bin/env python3
"""Cross-container HIP IPC event micro-benchmark — producer side.

Creates an interprocess torch.cuda.Event, records it on a stream, exports its
IPC handle, and writes the handle bytes to the shared /ipc volume so the
consumer (a separate container) can import it. Stays alive holding the event
until the consumer signals done.

Run via tests/ipc/event-compose.yaml in both HSA_ENABLE_IPC_MODE_LEGACY=1 and
unset to compare. Isolates the HIP event-IPC path from all of LMCache.
"""
import os, sys, time, torch

IPC = "/ipc"
LEG = os.environ.get("HSA_ENABLE_IPC_MODE_LEGACY", "<unset>")
print(f"[producer] torch={torch.__version__} hip={torch.version.hip} "
      f"LEGACY={LEG}", flush=True)

torch.cuda.init()
dev = torch.device("cuda", torch.cuda.current_device())
stream = torch.cuda.Stream(device=dev)
with torch.cuda.stream(stream):
    _ = torch.ones(4096, device=dev)          # real GPU work to order after
    evt = torch.cuda.Event(interprocess=True)
    evt.record(stream)

try:
    handle = evt.ipc_handle()
    print(f"[producer] ipc_handle() OK ({len(handle)} bytes)", flush=True)
except Exception as e:                          # noqa: BLE001
    print(f"[producer] ipc_handle() FAILED: {type(e).__name__}: {e}", flush=True)
    with open(f"{IPC}/producer.error", "w") as f:
        f.write(f"{type(e).__name__}: {e}")
    sys.exit(1)

with open(f"{IPC}/evt.handle", "wb") as f:
    f.write(handle)
with open(f"{IPC}/producer.ready", "w") as f:
    f.write("1")
print("[producer] handle written; holding event until consumer.done", flush=True)

for _ in range(120):
    if os.path.exists(f"{IPC}/consumer.done"):
        break
    time.sleep(1)
print("[producer] exiting", flush=True)
