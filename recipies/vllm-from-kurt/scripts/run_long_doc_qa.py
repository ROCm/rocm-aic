#!/usr/bin/env python3
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

"""Start vLLM + LMCache, wait until the OpenAI HTTP API is up, run long_doc_qa.

Intended for use inside the vllm-kurt container (paths under /app). Host
**`./vllm-container`** sets **`VLLM_CONTAINER_DATA_DIR`** (default **`/data`**
in-container) for LMCache and logs; set **`HF_HOME=/hf`** for Hub cache.
Place this script's options first; every other flag is forwarded to
``/app/LMCache/benchmarks/long_doc_qa/long_doc_qa.py`` (for example
``--hit-miss-ratio``, ``--repeat-mode``). You may insert a bare ``--`` between
the two groups if you prefer.

The served model defaults to ``meta-llama/Llama-3.1-8B-Instruct`` (override with
env ``VLLM_MODEL``). Unless you pass ``--model`` to ``long_doc_qa.py``, the same
default is injected so the client matches the server.

Examples::

    python3 /app/run_long_doc_qa.py --backend hipfile
    python3 /app/run_long_doc_qa.py --backend native --ready-timeout 7200 \\
        --num-documents 8 --hit-miss-ratio 3:1 --json-output
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

LONG_DOC_QA = "/app/LMCache/benchmarks/long_doc_qa/long_doc_qa.py"
AIS_STATS = "/app/hipFile/build/tools/ais-stats/ais-stats"


def _data_dir() -> str:
    """Host data mount inside the container (set by vllm-container)."""
    v = os.environ.get("VLLM_CONTAINER_DATA_DIR", "").strip()
    return v if v else "/data"


SERVER_LOG = os.path.join(_data_dir(), "server.txt")
# Dense Llama: avoids MXFP4 MoE on ROCm consumer GPUs. Match vllm-server-* /
# vllm-benchmark (override with env VLLM_MODEL).
_DEFAULT_VLLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _vllm_model() -> str:
    v = os.environ.get("VLLM_MODEL", "").strip()
    return v if v else _DEFAULT_VLLM_MODEL


def _rocr_visible_devices() -> str:
    v = os.environ.get("ROCR_VISIBLE_DEVICES", "").strip()
    return v if v else "0"


def _vllm_port(rocr: str) -> int:
    return int(f"800{rocr}")


def _lmcache_port(rocr: str) -> int:
    return int(f"699{rocr}")


def _lmcache_local_disk_uri() -> str:
    root = _data_dir().rstrip("/")
    return f"file://{root}/lmcache_test/"


def _lmcache_gds_dir() -> str:
    return os.path.join(_data_dir().rstrip("/"), "lmcache_gds") + os.sep


def _common_lmcache_env(lmcache_port: int) -> dict[str, str]:
    return {
        "LMCACHE_INTERNAL_API_SERVER_ENABLED": "true",
        "LMCACHE_INTERNAL_API_SERVER_PORT_START": str(lmcache_port),
        "LMCACHE_CHUNK_SIZE": "256",
        "LMCACHE_LOCAL_CPU": "False",
        "MIOPEN_USER_DB_PATH": "/app/miopen",
        "MIOPEN_FIND_MODE": "FAST",
        "VLLM_ROCM_USE_AITER": "1",
        "AMDGCN_USE_BUFFER_OPS": "0",
    }


def _vllm_serve_argv(vllm_port: int, *, load_format_dummy: bool) -> list[str]:
    out = [
        "vllm",
        "serve",
        _vllm_model(),
        "--host",
        "localhost",
        "--port",
        str(vllm_port),
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.90",
        "--block-size",
        "64",
        "--enable-prefix-caching",
        "--stream-interval",
        "20",
        "--max-num-batched-tokens",
        "8192",
        "--async-scheduling",
        "--attention-backend",
        "ROCM_AITER_UNIFIED_ATTN",
        "--kv-transfer-config",
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}',
    ]
    if load_format_dummy:
        out[3:3] = ["--load-format", "dummy"]
    return out


def _start_server_native(vllm_port: int, lmcache_port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(_common_lmcache_env(lmcache_port))
    env.update(
        {
            "LMCACHE_LOCAL_DISK": _lmcache_local_disk_uri(),
            "LMCACHE_MAX_LOCAL_DISK_SIZE": "1500.0",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": "48",
        }
    )
    argv = _vllm_serve_argv(vllm_port, load_format_dummy=False)
    log_f = open(SERVER_LOG, "ab", buffering=0)
    try:
        return subprocess.Popen(
            argv,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_f.close()


def _start_server_hipfile(vllm_port: int, lmcache_port: int) -> subprocess.Popen:
    gds = os.path.join(_data_dir(), "lmcache_gds")
    subprocess.run(["rm", "-fr", gds], check=False)
    env = os.environ.copy()
    env.update(_common_lmcache_env(lmcache_port))
    env.update(
        {
            "HIPFILE_UNSUPPORTED_FILE_SYSTEMS": "true",
            "HIPFILE_ALLOW_COMPAT_MODE": "false",
            "HIPFILE_STATS_LEVEL": "0",
            "LMCACHE_USE_GDS": "true",
            "LMCACHE_GDS_BACKEND": "hipfile",
            "LMCACHE_EXTRA_CONFIG": '{"gds_io_threads": 4}',
            "LMCACHE_GDS_PATH": _lmcache_gds_dir(),
            "LMCACHE_GDS_BUFFER_SIZE": "16384",
        }
    )
    argv = [AIS_STATS] + _vllm_serve_argv(vllm_port, load_format_dummy=True)
    log_f = open(SERVER_LOG, "ab", buffering=0)
    try:
        return subprocess.Popen(
            argv,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_f.close()


def wait_for_vllm_http(
    vllm_port: int, *, timeout_s: float, interval_s: float
) -> None:
    url = f"http://127.0.0.1:{vllm_port}/v1/models"
    deadline = time.monotonic() + timeout_s
    last_err: str | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last_err = repr(e)
        time.sleep(interval_s)
    raise TimeoutError(
        f"vLLM did not become ready at {url} within {timeout_s}s "
        f"(last error: {last_err}); see {SERVER_LOG}"
    )


def _terminate_process_group(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Launch vLLM, wait for /v1/models, run long_doc_qa benchmark."
    )
    ap.add_argument(
        "--backend",
        choices=("hipfile", "native"),
        default="hipfile",
        help="hipfile: dummy weights + GDS LMCache path (matches vllm-server-hipfile). "
        "native: real weights + on-disk LMCache (matches vllm-server-native-disk).",
    )
    ap.add_argument(
        "--ready-timeout",
        type=float,
        default=3600.0,
        help="Seconds to wait for GET /v1/models (default: 3600).",
    )
    ap.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between readiness polls (default: 5).",
    )
    ap.add_argument(
        "--skip-server",
        action="store_true",
        help="Do not start vLLM; only run long_doc_qa (server must already be running).",
    )
    args, bench_argv = ap.parse_known_args()
    bench_argv = [a for a in bench_argv if a != "--"]

    rocr = _rocr_visible_devices()
    vllm_port = _vllm_port(rocr)
    lmcache_port = _lmcache_port(rocr)

    if not any(a == "--port" for a in bench_argv):
        bench_argv = ["--port", str(vllm_port)] + bench_argv

    if "--model" not in bench_argv:
        bench_argv = ["--model", _vllm_model()] + bench_argv

    if not os.path.isfile(LONG_DOC_QA):
        print(f"ERROR: missing {LONG_DOC_QA} (wrong image or path).", file=sys.stderr)
        return 1

    server: subprocess.Popen | None = None
    try:
        if not args.skip_server:
            if args.backend == "native":
                server = _start_server_native(vllm_port, lmcache_port)
            else:
                if not os.path.isfile(AIS_STATS):
                    print(
                        f"ERROR: missing {AIS_STATS}; use --backend native "
                        "or rebuild the image.",
                        file=sys.stderr,
                    )
                    return 1
                server = _start_server_hipfile(vllm_port, lmcache_port)
            print(
                f"Started vLLM ({args.backend}) pid={server.pid}, "
                f"ROCR_VISIBLE_DEVICES={rocr}, "
                f"http://127.0.0.1:{vllm_port}/v1 (log: {SERVER_LOG})",
                flush=True,
            )
            wait_for_vllm_http(
                vllm_port,
                timeout_s=args.ready_timeout,
                interval_s=args.poll_interval,
            )
            print("vLLM HTTP API is up; running long_doc_qa.py", flush=True)

        rc = subprocess.call([sys.executable, LONG_DOC_QA] + bench_argv)
        return int(rc)
    finally:
        _terminate_process_group(server)


if __name__ == "__main__":
    raise SystemExit(main())
