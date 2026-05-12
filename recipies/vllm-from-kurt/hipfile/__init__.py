import ctypes
import os

_lib = ctypes.CDLL("libhipfile.so")

class HipFileError(ctypes.Structure):
    _fields_ = [("err", ctypes.c_int), ("hip_drv_err", ctypes.c_int)]

_lib.hipFileDriverOpen.argtypes = []
_lib.hipFileDriverOpen.restype = HipFileError
_lib.hipFileDriverClose.argtypes = []
_lib.hipFileDriverClose.restype = HipFileError
_lib.hipFileHandleRegister.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
_lib.hipFileHandleRegister.restype = HipFileError
_lib.hipFileHandleDeregister.argtypes = [ctypes.c_void_p]
_lib.hipFileHandleDeregister.restype = None
_lib.hipFileBufRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_lib.hipFileBufRegister.restype = HipFileError
_lib.hipFileBufDeregister.argtypes = [ctypes.c_void_p]
_lib.hipFileBufDeregister.restype = HipFileError
_lib.hipFileRead.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int64, ctypes.c_int64]
_lib.hipFileRead.restype = ctypes.c_ssize_t
_lib.hipFileWrite.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int64, ctypes.c_int64]
_lib.hipFileWrite.restype = ctypes.c_ssize_t

class _HandleUnion(ctypes.Union):
    _fields_ = [("fd", ctypes.c_int), ("hFile", ctypes.c_void_p)]

class _HipFileDescr(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("handle", _HandleUnion), ("fs_ops", ctypes.c_void_p)]

def _check_err(result, func_name="hipFile"):
    if result.err != 0:
        raise RuntimeError(f"{func_name} failed: err={result.err}, hip_drv_err={result.hip_drv_err}")

class CuFileDriver:
    def __init__(self):
        result = _lib.hipFileDriverOpen()
        _check_err(result, "hipFileDriverOpen")
    def close(self):
        _lib.hipFileDriverClose()
    def __del__(self):
        try: self.close()
        except: pass

class CuFile:
    def __init__(self, path, mode="r", use_direct_io=False):
        self.path, self.mode, self.use_direct_io = path, mode, use_direct_io
        self.fd, self.fh = None, ctypes.c_void_p()
    def __enter__(self):
        flags = os.O_RDONLY
        if self.mode in ("w", "w+", "r+"): flags = os.O_RDWR
        if "w" in self.mode: flags |= os.O_CREAT
        if self.use_direct_io: flags |= os.O_DIRECT
        self.fd = os.open(self.path, flags, 0o644)
        descr = _HipFileDescr()
        descr.type = 1
        descr.handle.fd = self.fd
        descr.fs_ops = None
        result = _lib.hipFileHandleRegister(ctypes.byref(self.fh), ctypes.byref(descr))
        if result.err != 0:
            os.close(self.fd)
            raise RuntimeError(f"hipFileHandleRegister failed: err={result.err}, hip_drv_err={result.hip_drv_err}")
        return self
    def __exit__(self, *args):
        if self.fh: _lib.hipFileHandleDeregister(self.fh)
        if self.fd is not None: os.close(self.fd)
    def read(self, buffer_ptr, size, file_offset=0, dev_offset=0):
        ptr = buffer_ptr if isinstance(buffer_ptr, ctypes.c_void_p) else ctypes.c_void_p(buffer_ptr)
        ret = _lib.hipFileRead(self.fh, ptr, size, file_offset, dev_offset)
        if ret < 0: raise RuntimeError(f"hipFileRead failed with error code {ret}")
        return ret
    def write(self, buffer_ptr, size, file_offset=0, dev_offset=0):
        ptr = buffer_ptr if isinstance(buffer_ptr, ctypes.c_void_p) else ctypes.c_void_p(buffer_ptr)
        ret = _lib.hipFileWrite(self.fh, ptr, size, file_offset, dev_offset)
        if ret < 0: raise RuntimeError(f"hipFileWrite failed with error code {ret}")
        return ret

from hipfile.bindings import hipFileBufRegister, hipFileBufDeregister
