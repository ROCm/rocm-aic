#!/usr/bin/env python3
"""Apply ROCm gfx1201 KV buffer IPC patches to LMCache source.

Cross-container zero-copy GPU IPC on gfx1201/RDNA4 via HIP VMM:
  hipMemCreate(hipMemHandleTypePosixFileDescriptor)
  hipMemExportToShareableHandle -> dmabuf fd
  Unix socket SCM_RIGHTS fd passing (/tmp/aic-ipc-fds.sock)
  hipMemImportFromShareableHandle on lmcache server side

Verified: 64 MB random data, SHA256 match, ~300 GB/s, cross-container.
See docker/lmcache/rocm_vmem_ipc.py for the wrapper implementation.

v0.5.6 migration notes:
  - lmcache/v1/platform/cuda/ renamed to lmcache/v1/platform/devices/cuda/
  - _select_ipc_wrapper_cls() / IPCPolicy now govern wrapper class selection;
    we hook it instead of monkey-patching CudaIPCWrapper.wrap().
  - DeviceIPCWrapper.Serialize / Deserialize use a closed msgspec allowlist;
    we extend it to accept RocmVmemIPCWrapper via pickle (its __getstate__).
  - vllm_multi_process_adapter.py create_transfer_context() call grew more
    args and layout_hints is fetched separately; anchor updated accordingly.
  - worker_transfer.py register call now uses _submit_registration(lambda:);
    anchor updated accordingly.
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

# 1. devices/cuda/__init__.py: hook _select_ipc_wrapper_cls to return
#    RocmVmemIPCWrapper on ROCm (gfx1201/RDNA4).  This replaces the v0.5.5
#    approach of monkey-patching CudaIPCWrapper.wrap() — v0.5.6 restructured
#    the platform package and introduced explicit class-selection machinery.
#    The function lives in lmcache/v1/platform/devices/cuda/__init__.py
#    (was lmcache/v1/platform/cuda/__init__.py in v0.5.5).
f1 = f'{BASE}/lmcache/v1/platform/devices/cuda/__init__.py'
patch(f1,
    '    return RawCudaIPCWrapper if policy.isolated_ipc else CudaIPCWrapper',
    '    if __import__("torch").version.hip:\n'
    '        try:\n'
    '            from rocm_vmem_ipc import RocmVmemIPCWrapper as _R\n'
    '            return _R\n'
    '        except ImportError:\n'
    '            raise RuntimeError(\n'
    '                "ROCm VMEM IPC module not found (rocm_vmem_ipc). "\n'
    '                "hipIpcGetMemHandle unavailable on gfx1201/RDNA4."\n'
    '            )\n'
    '    return RawCudaIPCWrapper if policy.isolated_ipc else CudaIPCWrapper',
    'devices/cuda: _select_ipc_wrapper_cls -> RocmVmemIPCWrapper on ROCm'
)

# 1b. DeviceIPCWrapper.Serialize/Deserialize: extend to handle
#     RocmVmemIPCWrapper, which uses pickle (__getstate__/__setstate__) rather
#     than the msgspec allowlist.  We wrap the existing staticmethods with
#     ROCm-aware versions that intercept instances of our wrapper first, then
#     fall through to the original logic for all other types.
#     This file is in lmcache/v1/platform/base/ipc_wrapper.py (unchanged path).
f1b = f'{BASE}/lmcache/v1/platform/base/ipc_wrapper.py'
# Inject at end of file (after all class definitions).
_ser_injection = (
    '\n'
    '# ROCm gfx1201: extend Serialize/Deserialize to handle RocmVmemIPCWrapper\n'
    '# (not in the upstream msgspec allowlist; uses pickle __getstate__).\n'
    'try:\n'
    '    import pickle as _pickle\n'
    '    from rocm_vmem_ipc import RocmVmemIPCWrapper as _RocmW\n'
    '    _orig_Serialize = DeviceIPCWrapper.__dict__["Serialize"].__func__\n'
    '    _orig_Deserialize = DeviceIPCWrapper.__dict__["Deserialize"].__func__\n'
    '\n'
    '    @staticmethod\n'
    '    def _rocm_Serialize(obj):\n'
    '        if type(obj) is _RocmW:\n'
    '            return b"\\x00rocm" + _pickle.dumps(obj.__getstate__(), protocol=4)\n'
    '        return _orig_Serialize(obj)\n'
    '\n'
    '    @staticmethod\n'
    '    def _rocm_Deserialize(data):\n'
    '        if data[:5] == b"\\x00rocm":\n'
    '            state = _pickle.loads(data[5:])\n'
    '            w = object.__new__(_RocmW)\n'
    '            w.__setstate__(state)\n'
    '            return w\n'
    '        return _orig_Deserialize(data)\n'
    '\n'
    '    DeviceIPCWrapper.Serialize = _rocm_Serialize\n'
    '    DeviceIPCWrapper.Deserialize = _rocm_Deserialize\n'
    '    print("  OK: DeviceIPCWrapper Serialize/Deserialize extended for RocmVmemIPCWrapper")\n'
    'except ImportError:\n'
    '    pass\n'
)
with open(f1b) as _f: _src1b = _f.read()
if 'rocm_vmem_ipc' not in _src1b:
    with open(f1b, 'w') as _f: _f.write(_src1b + _ser_injection)
    print('  OK: ipc_wrapper.py: Serialize/Deserialize extended for RocmVmemIPCWrapper')
else:
    print('  SKIP: ipc_wrapper.py already has RocmVmemIPCWrapper extension')

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

# 3. vllm_multi_process_adapter.py: skip registration on RuntimeError.
#    v0.5.6: create_transfer_context() now takes instance_id/req_client/mode
#    and layout_hints is fetched via self._layout_hints (not inline).
f3 = f'{BASE}/lmcache/integration/vllm/vllm_multi_process_adapter.py'
patch(f3,
    '        transfer_ctx = create_transfer_context(\n'
    '            kv_caches,\n'
    '            instance_id=self.instance_id,\n'
    '            req_client=self.req_client,\n'
    '            mode=self._mp_transfer_mode,\n'
    '        )\n'
    '        layout_hints = self._layout_hints',
    '        try:\n'
    '            transfer_ctx = create_transfer_context(\n'
    '                kv_caches,\n'
    '                instance_id=self.instance_id,\n'
    '                req_client=self.req_client,\n'
    '                mode=self._mp_transfer_mode,\n'
    '            )\n'
    '        except RuntimeError as _e:\n'
    '            import lmcache.logging as _ll\n'
    '            _ll.init_logger(__name__).warning("KV registration skipped: %s", _e)\n'
    '            return\n'
    '        layout_hints = self._layout_hints',
    'vllm_multi_process_adapter skips registration on ROCm (v0.5.6 anchor)'
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
# export_event body differs by LMCache version (v0.5.5 `return event.ipc_handle()`
# vs v0.5.6 `handle = event.ipc_handle(); ...; return handle`); patch whichever
# is present by prepending the producer-side sync + early return.
_exp_sync = (
    '        if __import__("torch").version.hip:\n'
    '            event.synchronize()  # type: ignore[attr-defined]  # producer-side sync\n'
    '            return b""\n'
)
_exp_variants = [
    ('        return event.ipc_handle()  # type: ignore[attr-defined]',
     _exp_sync + '        return event.ipc_handle()  # type: ignore[attr-defined]'),
    ('        handle = event.ipc_handle()  # type: ignore[attr-defined]',
     _exp_sync + '        handle = event.ipc_handle()  # type: ignore[attr-defined]'),
]
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
    _exp = next((p for p, _ in _exp_variants if p in src), None)
    if _exp is None and _imp_pat not in src:
        continue
    if _exp is not None:
        patch(f_event, _exp, dict(_exp_variants)[_exp],
              f'event_ipc.export_event: producer-side sync ({f_event})')
    else:
        print(f'  WARN: export_event pattern not found in {f_event}', file=sys.stderr)
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

# 6. worker_transfer.py: wrap kv_caches before passing to _submit_registration
#    so we can bail early if the wrapper list is empty (ROCm fallback to POSIX).
#    v0.5.6: the register call is now wrapped in _submit_registration(lambda:...)
#    with self._instance_id / self._req_client; anchor updated accordingly.
f4 = f'{BASE}/lmcache/v1/multiprocess/transfer_context/worker_transfer.py'
patch(f4,
    '        event_backend.check_event_support(device)\n'
    '\n'
    '        future = self._submit_registration(\n'
    '            lambda: self._req_client.register_kv_cache(\n'
    '                self._instance_id,\n'
    '                wrap_kv_caches(kv_caches),',
    '        event_backend.check_event_support(device)\n'
    '\n'
    '        _kv_wrappers = wrap_kv_caches(kv_caches)\n'
    '        if not _kv_wrappers:\n'
    '            import lmcache.logging as _ll\n'
    '            _ll.init_logger(__name__).warning(\n'
    '                "KV cache IPC registration skipped (no IPC wrappers). "\n'
    '                "L2 NVMe transfers will use NIXL POSIX."\n'
    '            )\n'
    '            self._device = device\n'
    '            self._event_backend = event_backend\n'
    '            return\n'
    '        future = self._submit_registration(\n'
    '            lambda: self._req_client.register_kv_cache(\n'
    '                self._instance_id,\n'
    '                _kv_wrappers,',
    'worker_transfer.register skips on empty wrappers (v0.5.6 anchor)'
)
