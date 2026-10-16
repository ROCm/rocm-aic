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

# export_event body differs across LMCache versions:
#   v0.5.5:  return event.ipc_handle()
#   v0.5.6:  handle = event.ipc_handle(); self._exported_events.append(event); return handle
# Patch whichever is present by prepending the producer-side sync + early return.
_SYNC = (
    '        if __import__("torch").version.hip:\n'
    '            event.synchronize()  # type: ignore[attr-defined]  # producer-side sync\n'
    '            return b""\n'
)
EXP_VARIANTS = [
    ('        return event.ipc_handle()  # type: ignore[attr-defined]',
     _SYNC + '        return event.ipc_handle()  # type: ignore[attr-defined]'),
    ('        handle = event.ipc_handle()  # type: ignore[attr-defined]',
     _SYNC + '        handle = event.ipc_handle()  # type: ignore[attr-defined]'),
]
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
    exp_pat = next((p for p, _ in EXP_VARIANTS if p in src), None)
    if exp_pat is None and IMP_PAT not in src:
        continue
    if exp_pat is not None:
        exp_new = dict(EXP_VARIANTS)[exp_pat]
        _apply(f_event, exp_pat, exp_new, f"export_event: producer-side sync ({f_event})")
    else:
        print(f"  WARN: export_event pattern not found in {f_event} (LMCache version drift?)",
              file=sys.stderr)
    _apply(f_event, IMP_PAT, IMP_NEW, f"import_event: dummy-complete event ({f_event})")
    done = True
    break
if not done:
    print("  WARN: event_ipc patterns not found in any known path; "
          "gfx1201 event producer-side-sync NOT applied", file=sys.stderr)
    sys.exit(1)
print("ROCm gfx1201 event-IPC producer-side-sync applied")
