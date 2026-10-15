#!/usr/bin/env python3
"""Apply ONLY the gfx1201 event-IPC producer-side-sync patch to LMCache.

Cross-process HIP event IPC is broken on gfx1201: hipIpcOpenEventHandle /
torch.cuda.Event.from_ipc_handle fails with hipErrorInvalidContext even
same-process, with the correct hipEventInterprocess|hipEventDisableTiming flags,
in both LEGACY and non-LEGACY modes (see tests/ipc/event_raw_hip.py). Memory IPC
(VMM dmabuf + SCM_RIGHTS) is unaffected.

So move the GPU-write ordering to the PRODUCER:
  export_event():  event.synchronize() on the producer's OWN context (works),
                   then return b"" — the write is complete before the request is
                   even sent.
  import_event():  never call the broken from_ipc_handle; return an
                   already-complete dummy event so wait_event()/query() are
                   no-ops and store()/retrieve() PROCEED (vs. patch 11's skip).

Both the vLLM client (producer on store) and the lmcache server (producer on
retrieve) must apply this. BASE (the dir containing `lmcache/`) defaults to the
source tree; set AIC_LMCACHE_SRC to patch a pip-installed package (vLLM image).

NOTE: this is also inlined as patch #4 of rocm_ipc_patch.py (the lmcache image
runs that). Keep the two in sync, or have the lmcache image call this script too.
"""
import os, sys

BASE = os.environ.get("AIC_LMCACHE_SRC", "/app/LMCache")

PATHS = [
    f"{BASE}/lmcache/v1/platform/base/event_ipc.py",
    f"{BASE}/lmcache/v1/multiprocess/event_ipc.py",
]

EXP_PAT = '        return event.ipc_handle()  # type: ignore[attr-defined]'
EXP_NEW = (
    '        if __import__("torch").version.hip:\n'
    '            event.synchronize()  # type: ignore[attr-defined]  # producer-side sync\n'
    '            return b""\n'
    '        return event.ipc_handle()  # type: ignore[attr-defined]'
)
IMP_PAT = (
    '        return self._event_module.Event.from_ipc_handle(  # type: ignore[union-attr]\n'
    '            device, handle\n'
    '        )'
)
IMP_NEW = (
    '        if __import__("torch").version.hip:\n'
    '            _ev = self._event_module.Event()  # type: ignore[union-attr]\n'
    '            _ev.record()\n'
    '            _ev.synchronize()\n'
    '            return _ev\n'
    '        return self._event_module.Event.from_ipc_handle(  # type: ignore[union-attr]\n'
    '            device, handle\n'
    '        )'
)


def _apply(path, old, new, name):
    with open(path) as f:
        src = f.read()
    if old in src:
        with open(path, "w") as f:
            f.write(src.replace(old, new, 1))
        print(f"  OK: {name}")
    elif new.strip().splitlines()[0] in src:
        print(f"  SKIP (already applied): {name}")
    else:
        print(f"  WARN: pattern not found: {name}", file=sys.stderr)


done = False
for f_event in PATHS:
    if not os.path.exists(f_event):
        continue
    src = open(f_event).read()
    if EXP_PAT in src or IMP_PAT in src:
        _apply(f_event, EXP_PAT, EXP_NEW, f"export_event: producer-side sync ({f_event})")
        _apply(f_event, IMP_PAT, IMP_NEW, f"import_event: dummy-complete event ({f_event})")
        done = True
        break
if not done:
    print("  WARN: event_ipc patterns not found in any known path; "
          "gfx1201 event producer-side-sync NOT applied", file=sys.stderr)
    sys.exit(1)
print("ROCm gfx1201 event-IPC producer-side-sync applied")
