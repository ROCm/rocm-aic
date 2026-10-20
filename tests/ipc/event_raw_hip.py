#!/usr/bin/env python3
"""Raw HIP event-IPC test via ctypes — bypasses torch's Event wrapper.

Creates the event with the REQUIRED flags
  hipEventInterprocess (4) | hipEventDisableTiming (2) = 6
then exercises hipIpcGetEventHandle -> hipIpcOpenEventHandle (same process).
This isolates whether the HIP event-IPC API itself works on gfx1201 vs whether
torch.cuda.Event(interprocess=True) fails to set the flags.

Run under both HSA_ENABLE_IPC_MODE_LEGACY settings.
"""
import ctypes, os

HIP_EVENT_DISABLE_TIMING = 2
HIP_EVENT_INTERPROCESS = 4
FLAGS = HIP_EVENT_INTERPROCESS | HIP_EVENT_DISABLE_TIMING

LEG = os.environ.get("HSA_ENABLE_IPC_MODE_LEGACY", "<unset>")
hip = ctypes.CDLL("libamdhip64.so")

class hipIpcEventHandle_t(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]

def name(rc):
    hip.hipGetErrorName.restype = ctypes.c_char_p
    return hip.hipGetErrorName(rc).decode()

print(f"[raw-hip] LEGACY={LEG} flags=hipEventInterprocess|DisableTiming({FLAGS})", flush=True)
hip.hipSetDevice(0)

evt = ctypes.c_void_p()
rc = hip.hipEventCreateWithFlags(ctypes.byref(evt), ctypes.c_uint(FLAGS))
print(f"[raw-hip] hipEventCreateWithFlags       rc={rc} ({name(rc)})", flush=True)

rc = hip.hipEventRecord(evt, ctypes.c_void_p(0))
print(f"[raw-hip] hipEventRecord                rc={rc} ({name(rc)})", flush=True)
hip.hipDeviceSynchronize()

h = hipIpcEventHandle_t()
rc = hip.hipIpcGetEventHandle(ctypes.byref(h), evt)
print(f"[raw-hip] hipIpcGetEventHandle          rc={rc} ({name(rc)})", flush=True)
get_ok = (rc == 0)

imp = ctypes.c_void_p()
rc = hip.hipIpcOpenEventHandle(ctypes.byref(imp), h)
open_ok = (rc == 0)
print(f"[raw-hip] hipIpcOpenEventHandle         rc={rc} ({name(rc)})", flush=True)

verdict = "WORKS" if (get_ok and open_ok) else "BROKEN"
print(f"[raw-hip] RESULT [{verdict}] get={get_ok} open={open_ok}", flush=True)
