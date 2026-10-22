#!/usr/bin/env python3
"""Diagnostic patch: instrument LMCacheMPWorkerAdapter to log when/why
transfer_ctx is None in create_recorded_event().

Adds:
  1. A traceback log in _send_register_kv_caches_request when transfer_ctx
     is successfully set, so we can confirm the registration happened and
     which call stack triggered it.
  2. A log + full traceback in create_recorded_event when transfer_ctx is
     None, so we can see the call chain that reaches the broken state.

Set AIC_LMCACHE_SRC to the directory containing lmcache/ when patching a
pip-installed package; defaults to the /app/LMCache source tree.
"""
import os
import sys

BASE = os.environ.get("AIC_LMCACHE_SRC", "/app/LMCache")
TARGET = os.path.join(BASE, "lmcache/integration/vllm/vllm_multi_process_adapter.py")

if not os.path.exists(TARGET):
    print(f"[diag-patch] NOT FOUND: {TARGET}", file=sys.stderr)
    sys.exit(1)

src = open(TARGET).read()
original = src

# ------------------------------------------------------------------
# Patch 1: instrument _send_register_kv_caches_request
# After   self.transfer_ctx = transfer_ctx
# Insert  a debug log with id() and a traceback so we know WHO called it.
# ------------------------------------------------------------------
OLD1 = "        self.transfer_ctx = transfer_ctx\n        try:"
NEW1 = """\
        self.transfer_ctx = transfer_ctx
        import traceback as _tb
        import logging as _log
        _log.getLogger(__name__).warning(
            "[diag] transfer_ctx SET id=%d on adapter id=%d\\n%s",
            id(transfer_ctx), id(self),
            "".join(_tb.format_stack()),
        )
        try:"""

if OLD1 not in src:
    print(f"[diag-patch] WARN: Patch 1 anchor not found in {TARGET}; "
          "source may differ from v0.5.6.", file=sys.stderr)
else:
    src = src.replace(OLD1, NEW1, 1)
    print(f"[diag-patch] Patch 1 applied: instrument transfer_ctx assignment")

# ------------------------------------------------------------------
# Patch 2: instrument create_recorded_event when transfer_ctx is None
# Replace the bare RuntimeError raise with a log-then-raise that includes
# the current stack so we can see what code path reaches this state.
# ------------------------------------------------------------------
OLD2 = '''\
        if transfer_ctx is None:
            raise RuntimeError(
                "KV caches are not registered. Call register_kv_caches() "
                "before creating transfer events."
            )'''
NEW2 = '''\
        if transfer_ctx is None:
            import traceback as _tb
            import logging as _log
            _log.getLogger(__name__).error(
                "[diag] create_recorded_event: transfer_ctx is None on adapter id=%d"
                "  (self.kv_caches set=%s)\\nStack:\\n%s",
                id(self),
                self.kv_caches is not None,
                "".join(_tb.format_stack()),
            )
            raise RuntimeError(
                "KV caches are not registered. Call register_kv_caches() "
                "before creating transfer events."
            )'''

if OLD2 not in src:
    print(f"[diag-patch] WARN: Patch 2 anchor not found in {TARGET}; "
          "source may differ from v0.5.6.", file=sys.stderr)
else:
    src = src.replace(OLD2, NEW2, 1)
    print(f"[diag-patch] Patch 2 applied: instrument create_recorded_event")

# ------------------------------------------------------------------
# Patch 3: instrument shutdown to log when it clears transfer_ctx
# ------------------------------------------------------------------
OLD3 = """\
        if self.transfer_ctx is not None:
            self.transfer_ctx.close()
            self.transfer_ctx = None"""
NEW3 = """\
        if self.transfer_ctx is not None:
            import traceback as _tb
            import logging as _log
            _log.getLogger(__name__).warning(
                "[diag] shutdown clearing transfer_ctx id=%d on adapter id=%d\\n%s",
                id(self.transfer_ctx), id(self),
                "".join(_tb.format_stack()),
            )
            self.transfer_ctx.close()
            self.transfer_ctx = None"""

if OLD3 not in src:
    print(f"[diag-patch] WARN: Patch 3 anchor not found in {TARGET}; "
          "source may differ from v0.5.6.", file=sys.stderr)
else:
    src = src.replace(OLD3, NEW3, 1)
    print(f"[diag-patch] Patch 3 applied: instrument shutdown")

if src == original:
    print("[diag-patch] ERROR: no patches applied — anchors not found", file=sys.stderr)
    sys.exit(1)

open(TARGET, "w").write(src)
print(f"[diag-patch] Done: {TARGET}")
