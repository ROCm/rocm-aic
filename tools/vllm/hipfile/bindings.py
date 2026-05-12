import ctypes
from hipfile import _lib, HipFileError, _check_err

def hipFileBufRegister(buffer_base, size, flags=0):
    result = _lib.hipFileBufRegister(buffer_base, size, flags)
    _check_err(result, "hipFileBufRegister")

def hipFileBufDeregister(buffer_base):
    result = _lib.hipFileBufDeregister(buffer_base)
    _check_err(result, "hipFileBufDeregister")
