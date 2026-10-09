#!/usr/bin/env python3
"""Apply ROCm gfx1201 KV buffer IPC patches to LMCache source.

Patches three files to gracefully handle hipIpcGetMemHandle being
unsupported on RDNA4 (gfx1201). Falls back to NIXL POSIX for L2 NVMe.
"""
import os, sys

BASE = '/app/LMCache'

def patch(path, old, new, name):
    with open(path) as f: src = f.read()
    if old in src:
        with open(path, 'w') as f: f.write(src.replace(old, new, 1))
        print(f'  OK: {name}')
    elif new.strip().split('\n')[0] in src:
        print(f'  SKIP (already applied): {name}')
    else:
        print(f'  WARN: pattern not found: {name}', file=sys.stderr)

# 1. ipc_wrapper.py: wrap() raises RuntimeError on ROCm so callers can skip
f1 = f'{BASE}/lmcache/v1/platform/cuda/ipc_wrapper.py'
with open(f1) as f: src = f.read()
wrap_start = src.find('    def wrap(cls, tensor')
wrap_end = src.find('\n    @', wrap_start + 1)
if wrap_start >= 0:
    section = src[wrap_start:wrap_end]
    old1 = '        return cls(tensor)'
    new1 = (
        '        if __import__("torch").version.hip:\n'
        '            raise RuntimeError(\n'
        '                "KV buffer IPC unsupported on ROCm gfx1201/RDNA4 "\n'
        '                "(hipIpcGetMemHandle not available). "\n'
        '                "L2 NVMe via NIXL POSIX will be used."\n'
        '            )\n'
        '        return cls(tensor)'
    )
    if old1 in section and 'hipIpcGetMemHandle' not in section:
        new_section = section.replace(old1, new1, 1)
        with open(f1, 'w') as f:
            f.write(src[:wrap_start] + new_section + src[wrap_end:])
        print('  OK: ipc_wrapper.wrap raises RuntimeError on ROCm')
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

# 4. event_ipc.py: export_event() returns b"" on ROCm so the importer skips
#    event synchronisation (hipIpcGetEventHandle also unsupported on gfx1201).
f_event = f'{BASE}/lmcache/v1/multiprocess/event_ipc.py'
if os.path.exists(f_event):
    patch(f_event,
        '        return event.ipc_handle()  # type: ignore[attr-defined]',
        '        if __import__("torch").version.hip:\n'
        '            return b""\n'
        '        return event.ipc_handle()  # type: ignore[attr-defined]',
        'event_ipc.export_event returns b"" on ROCm (hipIpcGetEventHandle unsupported)'
    )

# 5. worker_transfer.py: set device/event_backend before early return in register()
f4 = f'{BASE}/v1/multiprocess/transfer_context/worker_transfer.py'
patch(f4,
    '        future = req_client.register_kv_cache(\n'
    '            instance_id,\n'
    '            wrap_kv_caches(kv_caches),',
    '        kv_wrappers = wrap_kv_caches(kv_caches)\n'
    '        if not kv_wrappers:\n'
    '            import lmcache.logging as _ll\n'
    '            _ll.init_logger(__name__).warning(\n'
    '                "KV cache IPC registration skipped (no IPC wrappers, ROCm gfx1201/RDNA4). "\n'
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
