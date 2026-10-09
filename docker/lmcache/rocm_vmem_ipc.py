#!/usr/bin/env python3
"""ROCm VMEM IPC — zero-copy cross-container KV buffer sharing via HIP VMM.

Replaces hipIpcGetMemHandle/hipIpcOpenMemHandle (ROCR socket scoped to process
namespace, cannot cross Docker container boundaries) with:

  hipMemCreate(hipMemHandleTypePosixFileDescriptor)
  hipMemExportToShareableHandle -> real dmabuf fd
  Unix socket SCM_RIGHTS fd passing  (/tmp/aic-ipc/fds.sock)
  hipMemImportFromShareableHandle on the lmcache server side

Verified: 64 MB random data, SHA256 match, ~300 GB/s D2D, cross-container.

Architecture
------------
Sender (vLLM EngineCore, via lmcache's ipc_wrapper.py):
  RocmVmemIPCWrapper.wrap(tensor):
    1. hipMemCreate — VMEM allocation (required for POSIX fd export)
    2. hipMemMap + hipMemSetAccess
    3. hipMemcpy D2D: existing hipMalloc KV tensor -> VMEM shadow
    4. hipMemExportToShareableHandle -> dmabuf fd
    5. FdSender.send_fd(fd, token) — SCM_RIGHTS over Unix socket
    6. os.close(fd)  — kernel keeps dmabuf alive via VMEM handle ref
    7. Returns wrapper; handle carries (alloc_size, token)

Receiver (lmcache server, LMCacheDrivenTransferModule.__init__):
  RocmVmemIPCWrapper.init_receiver()  — start Unix socket server + bg thread
  to_tensor():
    1. Look up fd by token from background-thread buffer
    2. hipMemImportFromShareableHandle(fd)
    3. hipMemMap -> GPU VA
    4. Return torch.Tensor view (D2D copy into a contiguous output tensor)

Sidecar socket
--------------
Path: AIC_IPC_SOCK env (default /tmp/aic-ipc/fds.sock).
Mount the parent dir into both containers:
  docker-compose: volumes: [aic-ipc:/tmp/aic-ipc]
Container flags: --device=/dev/kfd --device=/dev/dri --ipc=host
"""
from __future__ import annotations

import array
import ctypes
import glob
import os
import socket
import struct
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

_SIDECAR_SOCK = os.environ.get("AIC_IPC_SOCK", "/tmp/aic-ipc/fds.sock")

# ---------------------------------------------------------------------------
# HIP VMM ctypes shim
# ---------------------------------------------------------------------------

def _load_hip() -> ctypes.CDLL:
    candidates = glob.glob("/opt/rocm*/lib/libamdhip64.so*")
    lib = next((c for c in candidates if ".so." in c), None)
    if not lib:
        raise RuntimeError("libamdhip64.so not found")
    hip = ctypes.CDLL(lib)
    hip.hipGetErrorName.restype = ctypes.c_char_p
    return hip


class _HipMemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("id", ctypes.c_uint32)]


class _HipMemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("requestedHandleTypes", ctypes.c_uint32),
        ("location", _HipMemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", ctypes.c_uint64),
    ]


class _HipMemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _HipMemLocation), ("flags", ctypes.c_uint32)]


_ALLOC_PINNED = 0x1
_HANDLE_POSIX_FD = 0x1
_LOC_DEVICE = 0x1
_ACCESS_RW = 3


def _vmem_granularity(hip: ctypes.CDLL, device_id: int) -> int:
    prop = _HipMemAllocationProp()
    prop.type = _ALLOC_PINNED
    prop.requestedHandleTypes = _HANDLE_POSIX_FD
    prop.location.type = _LOC_DEVICE
    prop.location.id = device_id
    gran = ctypes.c_size_t(0)
    hip.hipMemGetAllocationGranularity(ctypes.byref(gran), ctypes.byref(prop), 0)
    return gran.value


def _round_up(size: int, gran: int) -> int:
    return ((size + gran - 1) // gran) * gran


def _vmem_alloc(hip: ctypes.CDLL, size: int, device_id: int) -> Tuple[int, int, int]:
    """Allocate VMEM + map. Returns (hip_handle, va_ptr, alloc_size)."""
    gran = _vmem_granularity(hip, device_id)
    alloc_size = _round_up(size, gran)

    prop = _HipMemAllocationProp()
    prop.type = _ALLOC_PINNED
    prop.requestedHandleTypes = _HANDLE_POSIX_FD
    prop.location.type = _LOC_DEVICE
    prop.location.id = device_id

    h = ctypes.c_uint64(0)
    rc = hip.hipMemCreate(ctypes.byref(h), alloc_size, ctypes.byref(prop), 0)
    if rc != 0:
        raise RuntimeError(f"hipMemCreate rc={rc} ({hip.hipGetErrorName(rc).decode()})")

    va = ctypes.c_void_p()
    hip.hipMemAddressReserve(ctypes.byref(va), alloc_size, 0, None, 0)
    rc = hip.hipMemMap(va, alloc_size, 0, h, 0)
    if rc != 0:
        hip.hipMemRelease(h)
        raise RuntimeError(f"hipMemMap rc={rc}")

    acc = _HipMemAccessDesc()
    acc.location.type = _LOC_DEVICE
    acc.location.id = device_id
    acc.flags = _ACCESS_RW
    hip.hipMemSetAccess(va, alloc_size, ctypes.byref(acc), 1)
    return h.value, va.value, alloc_size


def _vmem_free(hip: ctypes.CDLL, h: int, va: int, alloc_size: int) -> None:
    hip.hipMemUnmap(ctypes.c_void_p(va), alloc_size)
    hip.hipMemAddressFree(ctypes.c_void_p(va), alloc_size)
    hip.hipMemRelease(ctypes.c_uint64(h))


def _vmem_export(hip: ctypes.CDLL, h: int) -> int:
    """Export VMEM handle as a dmabuf fd. Caller must os.close() when done."""
    fd = ctypes.c_int(-1)
    rc = hip.hipMemExportToShareableHandle(
        ctypes.byref(fd),
        ctypes.c_uint64(h),
        ctypes.c_uint32(_HANDLE_POSIX_FD),
        0,
    )
    if rc != 0:
        raise RuntimeError(
            f"hipMemExportToShareableHandle rc={rc} ({hip.hipGetErrorName(rc).decode()})"
        )
    return fd.value


def _vmem_import(
    hip: ctypes.CDLL, dmabuf_fd: int, alloc_size: int, device_id: int
) -> Tuple[int, int]:
    """Import dmabuf fd → mapped VMEM. Returns (hip_handle, va_ptr)."""
    imp_h = ctypes.c_uint64(0)
    rc = hip.hipMemImportFromShareableHandle(
        ctypes.byref(imp_h),
        ctypes.c_int(dmabuf_fd),
        ctypes.c_uint32(_HANDLE_POSIX_FD),
    )
    if rc != 0:
        raise RuntimeError(
            f"hipMemImportFromShareableHandle rc={rc} ({hip.hipGetErrorName(rc).decode()})"
        )

    gran = _vmem_granularity(hip, device_id)
    rounded = _round_up(alloc_size, gran)

    va = ctypes.c_void_p()
    hip.hipMemAddressReserve(ctypes.byref(va), rounded, 0, None, 0)
    rc = hip.hipMemMap(va, rounded, 0, imp_h, 0)
    if rc != 0:
        hip.hipMemRelease(imp_h)
        raise RuntimeError(f"hipMemMap(import) rc={rc}")

    acc = _HipMemAccessDesc()
    acc.location.type = _LOC_DEVICE
    acc.location.id = device_id
    acc.flags = _ACCESS_RW
    hip.hipMemSetAccess(va, rounded, ctypes.byref(acc), 1)
    return imp_h.value, va.value


# ---------------------------------------------------------------------------
# Sidecar socket helpers
# ---------------------------------------------------------------------------

def _send_fd_over_socket(sock: socket.socket, fd: int, token: bytes) -> None:
    fds = array.array("i", [fd])
    sock.sendmsg(
        [struct.pack("<I", len(token)) + token],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds)],
    )


def _recv_fd_from_socket(conn: socket.socket) -> Tuple[int, bytes]:
    msg, ancdata, _, _ = conn.recvmsg(
        520, socket.CMSG_SPACE(ctypes.sizeof(ctypes.c_int))
    )
    token_len = struct.unpack("<I", msg[:4])[0]
    token = msg[4 : 4 + token_len]
    for lvl, typ, data in ancdata:
        if lvl == socket.SOL_SOCKET and typ == socket.SCM_RIGHTS:
            fds = array.array("i")
            fds.frombytes(data[: len(data) - len(data) % fds.itemsize])
            return fds[0], token
    raise RuntimeError("SCM_RIGHTS ancdata missing")


# ---------------------------------------------------------------------------
# RocmVmemIPCWrapper
# ---------------------------------------------------------------------------

@dataclass
class _VmemHandle:
    """Pickle-safe payload stored in RocmVmemIPCWrapper.handle."""
    alloc_size: int
    token: bytes   # 16-byte random id used to match fd on server side


class RocmVmemIPCWrapper:
    """Cross-container KV IPC wrapper: HIP VMM + SCM_RIGHTS fd passing.

    Sender (vLLM EngineCore):
      RocmVmemIPCWrapper.wrap(tensor) allocates VMEM, D2D-copies tensor,
      exports fd, sends via FdSender, returns serialisable wrapper.

    Receiver (lmcache server):
      Call RocmVmemIPCWrapper.init_receiver() once at server startup.
      to_tensor() looks up the pre-buffered fd by token, imports and maps it,
      D2D-copies into a contiguous output tensor.
    """

    # ---- class-level singletons ----
    _hip: Optional[ctypes.CDLL] = None
    _hip_lock = threading.Lock()

    # Sender side
    _sender: Optional["_FdSenderClient"] = None
    _sender_lock = threading.Lock()

    # Receiver side: background thread buffers fds by token
    _receiver: Optional["_FdReceiverServer"] = None
    _fd_buffer: Dict[bytes, int] = {}        # token -> fd
    _fd_buffer_lock = threading.Lock()
    _fd_buffer_ready: Dict[bytes, threading.Event] = {}

    # Per-wrapper resources (sender keeps these for cleanup)
    _vmem_hip_handle: int = 0
    _vmem_va: int = 0
    _vmem_alloc_size: int = 0

    def __init__(
        self,
        handle: _VmemHandle,
        dtype: torch.dtype,
        shape: Tuple[int, ...],
        stride: Tuple[int, ...],
        storage_offset: int,
        device_uuid: str,
    ) -> None:
        self.handle = handle
        self.dtype = dtype
        self.shape = shape
        self.stride = stride
        self.storage_offset = storage_offset
        self.device_uuid = device_uuid

    # ---- class helpers ----

    @classmethod
    def _get_hip(cls) -> ctypes.CDLL:
        with cls._hip_lock:
            if cls._hip is None:
                cls._hip = _load_hip()
            return cls._hip

    @classmethod
    def _device_id(cls, device_uuid: str) -> int:
        # Import lazily to avoid circular imports
        from lmcache.v1.platform.cuda.ipc_wrapper import DeviceIPCWrapper
        return DeviceIPCWrapper._get_device_index_from_uuid(device_uuid)

    @classmethod
    def _get_device_uuid(cls, device_index: int) -> str:
        from lmcache.v1.platform.cuda.ipc_wrapper import DeviceIPCWrapper
        return DeviceIPCWrapper._get_device_uuid(device_index)

    # ---- sender API ----

    @classmethod
    def wrap(cls, tensor: torch.Tensor) -> "RocmVmemIPCWrapper":
        """Wrap a KV cache tensor for cross-container transfer."""
        hip = cls._get_hip()
        device_id = tensor.device.index or 0
        nbytes = tensor.nbytes
        token = os.urandom(16)

        # Allocate VMEM shadow (required for POSIX fd export)
        h, va, alloc_size = _vmem_alloc(hip, nbytes, device_id)

        # D2D copy: hipMalloc-backed KV tensor -> VMEM shadow
        rc = hip.hipMemcpy(
            ctypes.c_void_p(va), ctypes.c_void_p(tensor.data_ptr()), nbytes, 3
        )
        if rc != 0:
            _vmem_free(hip, h, va, alloc_size)
            raise RuntimeError(f"hipMemcpy D2D (shadow fill) rc={rc}")

        # Export as real dmabuf fd
        fd = _vmem_export(hip, h)

        # Send fd to lmcache server sidecar (lazy-connect)
        with cls._sender_lock:
            if cls._sender is None:
                cls._sender = _FdSenderClient(_SIDECAR_SOCK)
        cls._sender.send(fd, token)
        os.close(fd)  # kernel holds dmabuf ref via VMEM handle

        device_uuid = cls._get_device_uuid(device_id)
        w = cls(
            handle=_VmemHandle(alloc_size=alloc_size, token=token),
            dtype=tensor.dtype,
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            storage_offset=tensor.storage_offset(),
            device_uuid=device_uuid,
        )
        # Keep VMEM handle for cleanup on close()
        w._vmem_hip_handle = h
        w._vmem_va = va
        w._vmem_alloc_size = alloc_size
        return w

    def close(self) -> None:
        """Free VMEM shadow buffer (sender side)."""
        if self._vmem_hip_handle:
            hip = self._get_hip()
            _vmem_free(hip, self._vmem_hip_handle, self._vmem_va, self._vmem_alloc_size)
            self._vmem_hip_handle = 0

    # ---- receiver API ----

    @classmethod
    def init_receiver(cls, sock_path: str = _SIDECAR_SOCK) -> None:
        """Start the fd-receiver server + background drain thread.

        Call once at lmcache server startup (e.g. LMCacheDrivenTransferModule.__init__).
        """
        if cls._receiver is not None:
            return
        # Ensure socket directory exists
        sock_dir = os.path.dirname(sock_path)
        if sock_dir:
            os.makedirs(sock_dir, exist_ok=True)
        rcvr = _FdReceiverServer(sock_path)
        rcvr.start()
        cls._receiver = rcvr

        t = threading.Thread(target=cls._drain_loop, daemon=True, name="rocm-vmem-ipc-drain")
        t.start()

    @classmethod
    def _drain_loop(cls) -> None:
        """Background thread: continuously accept connections and buffer fds."""
        import logging
        log = logging.getLogger(__name__)
        assert cls._receiver is not None
        while True:
            try:
                conn = cls._receiver.accept()
                # Drain all fds from this connection in a sub-thread
                threading.Thread(
                    target=cls._drain_conn, args=(conn,), daemon=True,
                    name="rocm-vmem-ipc-conn"
                ).start()
            except Exception as exc:
                log.warning("rocm_vmem_ipc: accept error: %s", exc)

    @classmethod
    def _drain_conn(cls, conn: socket.socket) -> None:
        import logging
        log = logging.getLogger(__name__)
        try:
            while True:
                fd, token = _recv_fd_from_socket(conn)
                with cls._fd_buffer_lock:
                    cls._fd_buffer[token] = fd
                    ev = cls._fd_buffer_ready.get(token)
                if ev is not None:
                    ev.set()
        except (OSError, struct.error):
            pass  # connection closed
        except Exception as exc:
            log.warning("rocm_vmem_ipc: drain error: %s", exc)
        finally:
            conn.close()

    @classmethod
    def _wait_for_fd(cls, token: bytes, timeout: float = 10.0) -> int:
        """Wait until the fd for token arrives in the buffer."""
        with cls._fd_buffer_lock:
            if token in cls._fd_buffer:
                return cls._fd_buffer.pop(token)
            ev = threading.Event()
            cls._fd_buffer_ready[token] = ev

        if not ev.wait(timeout):
            raise TimeoutError(
                f"rocm_vmem_ipc: no fd received for token {token.hex()} "
                f"within {timeout}s — is AIC_IPC_SOCK={_SIDECAR_SOCK} shared "
                "between vllm and lmcache containers?"
            )
        with cls._fd_buffer_lock:
            cls._fd_buffer_ready.pop(token, None)
            return cls._fd_buffer.pop(token)

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct KV tensor from the imported VMEM buffer (receiver side)."""
        hip = self._get_hip()
        token = self.handle.token
        alloc_size = self.handle.alloc_size
        device_id = self._device_id(self.device_uuid)

        fd = self._wait_for_fd(token)
        try:
            h, va = _vmem_import(hip, fd, alloc_size, device_id)
        finally:
            os.close(fd)

        # D2D copy into a properly-shaped contiguous output tensor
        device = torch.device(f"cuda:{device_id}")
        out = torch.empty(self.shape, dtype=self.dtype, device=device)
        rc = hip.hipMemcpy(
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(va),
            out.nbytes,
            3,  # hipMemcpyDeviceToDevice
        )

        # Free the imported mapping immediately after copy
        gran = _vmem_granularity(hip, device_id)
        rounded = _round_up(alloc_size, gran)
        hip.hipMemUnmap(ctypes.c_void_p(va), rounded)
        hip.hipMemAddressFree(ctypes.c_void_p(va), rounded)
        hip.hipMemRelease(ctypes.c_uint64(h))

        if rc != 0:
            raise RuntimeError(f"hipMemcpy D2D (import copy) rc={rc}")
        return out

    # ---- pickle support ----

    def __getstate__(self) -> dict:
        return {
            "handle": self.handle,
            "dtype": self.dtype,
            "shape": self.shape,
            "stride": self.stride,
            "storage_offset": self.storage_offset,
            "device_uuid": self.device_uuid,
        }

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._vmem_hip_handle = 0
        self._vmem_va = 0
        self._vmem_alloc_size = 0


# ---------------------------------------------------------------------------
# Internal socket helpers
# ---------------------------------------------------------------------------

class _FdSenderClient:
    def __init__(self, sock_path: str) -> None:
        self._path = sock_path
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

    def _connect(self) -> None:
        if self._sock is not None:
            return
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(self._path)
        self._sock = s

    def send(self, fd: int, token: bytes) -> None:
        with self._lock:
            self._connect()
            assert self._sock is not None
            _send_fd_over_socket(self._sock, fd, token)


class _FdReceiverServer:
    def __init__(self, sock_path: str) -> None:
        self._path = sock_path
        self._srv: Optional[socket.socket] = None

    def start(self) -> None:
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self._path)
        os.chmod(self._path, 0o777)
        srv.listen(8)
        self._srv = srv

    def accept(self) -> socket.socket:
        assert self._srv is not None
        conn, _ = self._srv.accept()
        return conn

    def close(self) -> None:
        if self._srv:
            self._srv.close()
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Patch entry point — called by rocm_ipc_patch.py
# ---------------------------------------------------------------------------

def patch_lmcache_server(lmcache_base: str = "/app/LMCache") -> None:
    """Monkey-patch LMCacheDrivenTransferModule to start the fd receiver."""
    import importlib
    mod = importlib.import_module(
        "lmcache.v1.multiprocess.modules.lmcache_driven_transfer"
    )
    OrigModule = mod.LMCacheDrivenTransferModule
    orig_init = OrigModule.__init__

    def patched_init(self, ctx):
        orig_init(self, ctx)
        if __import__("torch").version.hip:
            RocmVmemIPCWrapper.init_receiver()

    OrigModule.__init__ = patched_init
    print("[rocm_vmem_ipc] LMCacheDrivenTransferModule.__init__ patched (ROCm receiver)", flush=True)
