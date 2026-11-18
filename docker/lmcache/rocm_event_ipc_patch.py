#!/usr/bin/env python3
"""Apply the ROCm event-IPC producer-side-sync edit to LMCache.

This is the only AIC source rewrite of LMCache outside patches/lmcache/; it
touches one file, lmcache/v1/platform/base/event_ipc.py.

Cross-process HIP event IPC is broken on gfx1201: hipIpcOpenEventHandle /
torch.cuda.Event.from_ipc_handle fails with hipErrorInvalidContext even
same-process, with the correct hipEventInterprocess|hipEventDisableTiming flags,
in both LEGACY and non-LEGACY modes (see tests/ipc/event_raw_hip.py). Memory IPC
is unaffected and stays LMCache's stock CudaIPCWrapper (hipIpcGetMemHandle).

So move the GPU-write ordering to the PRODUCER:
  export_event():  event.synchronize() on the producer's OWN context (works),
                   then return b"" — the write is complete before the request is
                   even sent.
  import_event():  never call the broken from_ipc_handle; return an
                   already-complete dummy event so wait_event()/query() are
                   no-ops and store()/retrieve() PROCEED.

Both the vLLM client (producer on store) and the lmcache server (producer on
retrieve) need it, so every LMCache that ships in an AIC image is patched by
this script:
  docker/lmcache/Dockerfile      the source tree, before the wheel is built
                                 (the wheel the default aic-vllm also installs)
  docker/vllm/Dockerfile         the PyPI client of the `emulate` stage
  docker/wheels-test/Dockerfile  the installed package, in both images

AIC_LMCACHE_SRC is the directory that CONTAINS `lmcache/`: a source checkout
or site-packages.  It defaults to /app/LMCache, the checkout in the lmcache
image's wheel-build stage.

Both replacements must apply or be found already applied; otherwise the script
exits non-zero, so an LMCache bump that moves an anchor stops the image build
instead of shipping an unpatched package.  Re-running on a patched tree changes
nothing and exits 0.
"""
import os, sys

BASE = os.environ.get('AIC_LMCACHE_SRC', '/app/LMCache')

_errors = []


def _fail(msg):
    _errors.append(msg)
    print(f'  ERROR: {msg}', file=sys.stderr)


def patch(path, old, new, name):
    """Replace the single occurrence of `old` in `path` with `new`.

    `new` is looked for first: the export edit prepends to its anchor, so the
    anchor is still there once the edit is in.  An anchor that is missing or
    not unique is an error.
    """
    try:
        with open(path) as f:
            src = f.read()
    except OSError as e:
        _fail(f'{name}: cannot read {path}: {e}')
        return
    if new in src:
        print(f'  SKIP (already applied): {name}')
        return
    if src.count(old) != 1:
        _fail(f'{name}: anchor found {src.count(old)} times, expected 1, in '
              f'{path} (LMCache version drift?)')
        return
    with open(path, 'w') as f:
        f.write(src.replace(old, new, 1))
    print(f'  OK: {name}')


# Event IPC — PRODUCER-SIDE SYNC (lmcache/v1/platform/base/event_ipc.py,
# DefaultEventIPCBackend; the file is unchanged from v0.5.6 to v0.5.8rc1, and
# RocmDeviceSpec still inherits this backend from CudaDeviceSpec).
#   export_event():  synchronize the event on the producer's own context, then
#                    return b"" — the GPU work is complete before the request
#                    or the response is sent.
#   import_event():  never call from_ipc_handle; return an already-complete
#                    event so wait_event()/query_event()/synchronize_event()
#                    are no-ops and store()/retrieve() proceed.
# Every export and import goes through these two methods: the client's
# submit_store/submit_retrieve (lmcache/v1/client/worker_transfer.py), the
# server's store/retrieve handlers, and the client's DeviceMessagingFuture
# (lmcache/v1/transport/futures.py, which took over from
# lmcache/v1/multiprocess/futures.py).  That future imports nothing when the
# handle is empty, which is what a patched server returns; a patched client
# would get the dummy event from a non-empty one.
# Not covered: the timeline-semaphore backend that --isolated-ipc selects.
_EVENT_IPC = 'lmcache/v1/platform/base/event_ipc.py'
_EXPORT_OLD = '        handle = event.ipc_handle()  # type: ignore[attr-defined]\n'
_EXPORT_NEW = (
    '        if __import__("torch").version.hip:\n'
    '            event.synchronize()  # type: ignore[attr-defined]  # producer-side sync\n'
    '            return b""\n'
) + _EXPORT_OLD
_IMPORT_OLD = (
    '        return self._event_module.Event.from_ipc_handle(  # type: ignore[union-attr]\n'
    '            device, handle\n'
    '        )\n'
)
_IMPORT_NEW = (
    '        if __import__("torch").version.hip:\n'
    '            _ev = self._event_module.Event()  # type: ignore[union-attr]\n'
    '            _ev.record()\n'
    '            _ev.synchronize()\n'
    '            return _ev\n'
) + _IMPORT_OLD


def patch_event_ipc(base):
    path = f'{base}/{_EVENT_IPC}'
    patch(path, _EXPORT_OLD, _EXPORT_NEW,
          f'event_ipc.export_event: producer-side sync ({path})')
    patch(path, _IMPORT_OLD, _IMPORT_NEW,
          f'event_ipc.import_event: dummy-complete event ({path})')


patch_event_ipc(BASE)

if _errors:
    print(f'ROCm event-IPC producer-side-sync NOT applied under {BASE}: '
          f'{len(_errors)} edit(s) failed', file=sys.stderr)
    sys.exit(1)
print('ROCm event-IPC producer-side-sync applied')
