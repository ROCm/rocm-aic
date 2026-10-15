#!/usr/bin/env python3
"""Apply ROCm gfx1201 KV buffer IPC patches to LMCache source.

Cross-container zero-copy GPU IPC on gfx1201/RDNA4 via HIP VMM:
  hipMemCreate(hipMemHandleTypePosixFileDescriptor)
  hipMemExportToShareableHandle -> dmabuf fd
  Unix socket SCM_RIGHTS fd passing (/tmp/aic-ipc-fds.sock)
  hipMemImportFromShareableHandle on lmcache server side

Verified: 64 MB random data, SHA256 match, ~300 GB/s, cross-container.
See docker/lmcache/rocm_vmem_ipc.py for the wrapper implementation.
"""
import os, sys, shutil

# Root that CONTAINS the `lmcache/` package. Defaults to the source tree the
# lmcache image patches before `setup.py bdist_wheel`; override with
# AIC_LMCACHE_SRC to patch a pip-installed package in site-packages (e.g. the
# vLLM image, which installs the lmcache client via `pip install lmcache`).
BASE = os.environ.get('AIC_LMCACHE_SRC', '/app/LMCache')
VMEM_MODULE_SRC = '/tmp/rocm_vmem_ipc.py'   # COPYed into image by Dockerfile
# Install to site-packages so it survives the `rm -rf /app/LMCache` cleanup step.
import sysconfig as _sc
_site = _sc.get_path('purelib')
VMEM_MODULE_DST = f'{_site}/rocm_vmem_ipc.py'

def patch(path, old, new, name):
    with open(path) as f: src = f.read()
    if old in src:
        with open(path, 'w') as f: f.write(src.replace(old, new, 1))
        print(f'  OK: {name}')
    elif new.strip().split('\n')[0] in src:
        print(f'  SKIP (already applied): {name}')
    else:
        print(f'  WARN: pattern not found: {name}', file=sys.stderr)

# Install the VMEM IPC module alongside the LMCache source so it can be imported.
if os.path.exists(VMEM_MODULE_SRC):
    shutil.copy(VMEM_MODULE_SRC, VMEM_MODULE_DST)
    print(f'  OK: installed {VMEM_MODULE_DST}')
else:
    print(f'  WARN: {VMEM_MODULE_SRC} not found — VMEM IPC unavailable', file=sys.stderr)

# 1. ipc_wrapper.py: wrap() redirects to RocmVmemIPCWrapper on ROCm.
#    Falls back to RuntimeError if rocm_vmem_ipc is unavailable (e.g. CDNA).
f1 = f'{BASE}/lmcache/v1/platform/cuda/ipc_wrapper.py'
with open(f1) as f: src = f.read()
wrap_start = src.find('    def wrap(cls, tensor')
wrap_end = src.find('\n    @', wrap_start + 1)
if wrap_start >= 0:
    section = src[wrap_start:wrap_end]
    old1 = '        return cls(tensor)'
    new1 = (
        '        if __import__("torch").version.hip:\n'
        '            try:\n'
        '                from rocm_vmem_ipc import RocmVmemIPCWrapper as _R\n'
        '                return _R.wrap(tensor)\n'
        '            except ImportError:\n'
        '                raise RuntimeError(\n'
        '                    "ROCm VMEM IPC module not found (rocm_vmem_ipc). "\n'
        '                    "hipIpcGetMemHandle unavailable on gfx1201/RDNA4."\n'
        '                )\n'
        '        return cls(tensor)'
    )
    if old1 in section and 'RocmVmemIPCWrapper' not in section:
        new_section = section.replace(old1, new1, 1)
        with open(f1, 'w') as f:
            f.write(src[:wrap_start] + new_section + src[wrap_end:])
        print('  OK: ipc_wrapper.wrap -> RocmVmemIPCWrapper on ROCm')
    else:
        print('  SKIP: ipc_wrapper already patched')

# 2. kv_wrap.py: return [] on RuntimeError from wrap_one_kv_cache loop
f2 = f'{BASE}/lmcache/v1/platform/kv_wrap.py'
patch(f2,
    '    except BaseException:\n        _release_partial_kv_wrappers(wrappers)\n        raise\n    return wrappers',
    '    except RuntimeError as _e:\n'
    '        logger.warning("KV IPC unavailable (%s); NIXL POSIX L2 will be used.", _e)\n'
    '        _release_partial_kv_wrappers(wrappers)\n'
    '        return []\n'
    '    except BaseException:\n'
    '        _release_partial_kv_wrappers(wrappers)\n'
    '        raise\n'
    '    return wrappers',
    'kv_wrap.wrap_kv_caches returns [] on ROCm'
)

# 3. vllm_multi_process_adapter.py: skip registration on RuntimeError
f3 = f'{BASE}/lmcache/integration/vllm/vllm_multi_process_adapter.py'
patch(f3,
    '        transfer_ctx = create_transfer_context(kv_caches, mode=self._mp_transfer_mode)\n'
    '        layout_hints',
    '        try:\n'
    '            transfer_ctx = create_transfer_context(kv_caches, mode=self._mp_transfer_mode)\n'
    '        except RuntimeError as _e:\n'
    '            import lmcache.logging as _ll\n'
    '            _ll.init_logger(__name__).warning("KV registration skipped: %s", _e)\n'
    '            return\n'
    '        layout_hints',
    'vllm_multi_process_adapter skips registration on ROCm'
)

print('ROCm KV IPC patches done — part 1')

# 4. event_ipc.py — PRODUCER-SIDE SYNC for gfx1201.
#    Cross-process HIP event IPC is broken on gfx1201: torch.cuda.Event
#    .from_ipc_handle raises hipErrorInvalidContext (verified same-process AND
#    cross-container, LEGACY and non-LEGACY — see tests/ipc/).  Memory IPC
#    (VMM dmabuf + SCM_RIGHTS) works fine; only the event-based ordering is
#    unavailable.  So move the ordering to the PRODUCER:
#      export_event():  event.synchronize() on the producer's OWN context (works),
#                       then return b"" — the GPU write is complete before the
#                       store/retrieve request is even sent.
#      import_event():  never call the broken from_ipc_handle; return an
#                       already-complete dummy event so wait_event()/query() are
#                       no-ops and store()/retrieve() proceed (vs. patch 11's skip).
#    Correct: the producer guaranteed the write landed before signalling.  BOTH
#    the vLLM client AND the lmcache server images must apply this (vLLM image
#    currently does NOT run this script — see docker/vllm/Dockerfile).
#    The multi-path loop tolerates event_ipc.py moving across LMCache versions.
_event_paths = [
    f'{BASE}/lmcache/v1/platform/base/event_ipc.py',
    f'{BASE}/lmcache/v1/multiprocess/event_ipc.py',
]
_exp_pat = '        return event.ipc_handle()  # type: ignore[attr-defined]'
_exp_new = (
    '        if __import__("torch").version.hip:\n'
    '            event.synchronize()  # type: ignore[attr-defined]  # producer-side sync\n'
    '            return b""\n'
    '        return event.ipc_handle()  # type: ignore[attr-defined]'
)
_imp_pat = (
    '        return self._event_module.Event.from_ipc_handle(  # type: ignore[union-attr]\n'
    '            device, handle\n'
    '        )'
)
_imp_new = (
    '        if __import__("torch").version.hip:\n'
    '            _ev = self._event_module.Event()  # type: ignore[union-attr]\n'
    '            _ev.record()\n'
    '            _ev.synchronize()\n'
    '            return _ev\n'
    '        return self._event_module.Event.from_ipc_handle(  # type: ignore[union-attr]\n'
    '            device, handle\n'
    '        )'
)
_event_done = False
for f_event in _event_paths:
    if not os.path.exists(f_event):
        continue
    src = open(f_event).read()
    if _exp_pat in src or _imp_pat in src:
        patch(f_event, _exp_pat, _exp_new,
              f'event_ipc.export_event: producer-side sync ({f_event})')
        patch(f_event, _imp_pat, _imp_new,
              f'event_ipc.import_event: dummy-complete event ({f_event})')
        _event_done = True
        break
if not _event_done:
    print('  WARN: event_ipc patterns not found; producer-side-sync NOT applied '
          '(gfx1201 event-IPC)', file=sys.stderr)

# 5. lmcache_driven_transfer.py: start VMEM IPC fd-receiver at server startup.
#    Injects an import + init_receiver() call into __init__ so the background
#    thread is running before any register_kv_cache request arrives.
f5 = f'{BASE}/lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py'
patch(f5,
    '        self._device_host_func_dispatcher = DeviceHostFuncDispatcher()',
    '        if __import__("torch").version.hip:\n'
    '            try:\n'
    '                from rocm_vmem_ipc import RocmVmemIPCWrapper as _R\n'
    '                _R.init_receiver()\n'
    '            except Exception as _e:\n'
    '                import warnings\n'
    '                warnings.warn(f"ROCm VMEM IPC receiver init failed: {_e}")\n'
    '        self._device_host_func_dispatcher = DeviceHostFuncDispatcher()',
    'lmcache_driven_transfer: start VMEM IPC receiver at server init'
)

# 6. worker_transfer.py: set _device/_event_backend before early return when
#    kv_wrappers is empty. Kept as a safety net — if RocmVmemIPCWrapper.wrap()
#    raises, wrap_kv_caches returns [] and this prevents AttributeError downstream.
f4 = f'{BASE}/lmcache/v1/multiprocess/transfer_context/worker_transfer.py'
patch(f4,
    '        future = req_client.register_kv_cache(\n'
    '            instance_id,\n'
    '            wrap_kv_caches(kv_caches),',
    '        kv_wrappers = wrap_kv_caches(kv_caches)\n'
    '        if not kv_wrappers:\n'
    '            import lmcache.logging as _ll\n'
    '            _ll.init_logger(__name__).warning(\n'
    '                "KV cache IPC registration skipped (no IPC wrappers). "\n'
    '                "L2 NVMe transfers will use NIXL POSIX."\n'
    '            )\n'
    '            self._device = device\n'
    '            self._event_backend = event_backend\n'
    '            return\n'
    '        future = req_client.register_kv_cache(\n'
    '            instance_id,\n'
    '            kv_wrappers,',
    'worker_transfer.register skips on empty wrappers, sets device/event_backend'
)
