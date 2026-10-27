#!/usr/bin/env python3
# WSL2 librocmdxg shim for vLLM.
# The fake amdsmi.py is injected via PYTHONPATH=/wsl2-stubs in GPU_ARGS so
# all processes (including EngineCore subprocess) get ROCm platform detection.
# LD_PRELOAD loads libdxcore.so + librocdxg.so at C level for all processes.
# This wrapper just loads the GPU libs at Python level for belt-and-suspenders,
# then hands off to the standard vLLM OpenAI entrypoint via runpy.
import ctypes, runpy, sys

for lib in ["libdxcore.so", "librocdxg.so"]:
    try:
        ctypes.CDLL(lib)
    except OSError:
        pass  # LD_PRELOAD already loaded them; ctypes is belt-and-suspenders

sys.argv[0] = "vllm.entrypoints.openai.api_server"
runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__", alter_sys=True)
