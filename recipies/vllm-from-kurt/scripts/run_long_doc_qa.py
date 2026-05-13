#!/usr/bin/env python3
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

"""Start vLLM + LMCache, wait until the OpenAI HTTP API is up, run long_doc_qa.

Intended for use inside the vllm-kurt container (paths under /app). Host
**`./vllm-container`** sets **`KURT_CONTAINER_DATA_DIR`** (default **`/data`**
in-container) for LMCache and logs; set **`HF_HOME=/hf`** for Hub cache.
Place this script's options first; every other flag is forwarded to
``/app/LMCache/benchmarks/long_doc_qa/long_doc_qa.py`` (for example
``--hit-miss-ratio``, ``--repeat-mode``). You may insert a bare ``--`` between
the two groups if you prefer.

The served model defaults to ``meta-llama/Llama-3.1-8B-Instruct`` (override with
env ``VLLM_MODEL``). Unless you pass ``--model`` to ``long_doc_qa.py``, the same
default is injected so the client matches the server. VRAM knobs use the
**`KURT_*`** prefix (not **`VLLM_*`**) so vLLM does not warn about unknown env
vars: **`KURT_GPU_MEMORY_UTILIZATION`** (default ``0.72``), **`KURT_MAX_MODEL_LEN`**
(default ``8192``), **`KURT_MAX_NUM_BATCHED_TOKENS`** (default ``4096``).
**`--enforce-eager`** is on unless **`KURT_ENFORCE_EAGER=0`** (legacy:
**`VLLM_ENFORCE_EAGER`**). Optional: **`KURT_PYTORCH_ALLOC_CONF`** or
**`VLLM_PYTORCH_ALLOC_CONF`** for **`PYTORCH_ALLOC_CONF`**.

LMCache observability (defaults favor Slurm report artifacts under
**`${KURT_CONTAINER_DATA_DIR}`**):

- **`KURT_LMCACHE_ENABLE_CHUNK_STATISTICS`**: ``1`` (default) enables chunk
  statistics (``file_hash`` strategy by default) and sets **`PYTHONHASHSEED=0`**
  for stable Bloom filters when using **`memory_bloom_filter`**.
- **`KURT_LMCACHE_CHUNK_STATISTICS_STRATEGY`**: ``file_hash`` (default) or
  ``memory_bloom_filter``.
- **`KURT_LMCACHE_COLLECT_INTERNAL_API`**: ``1`` (default) snapshots
  ``/chunk_statistics/status`` and ``/metrics`` from internal API ports before
  the server exits (written as ``lmcache_internal_api_*`` under the data dir).
- **`KURT_LMCACHE_INTERNAL_API_MAX_OFFSET`**: highest port index to probe
  relative to **LMCACHE_INTERNAL_API_SERVER_PORT_START** (default ``3``).
- **`KURT_LMCACHE_ENABLE_KV_EVENTS`**: ``1`` (default) sets
  **`LMCACHE_ENABLE_KV_EVENTS=true`** for LMCache KV connector events; set ``0``
  to disable.
- **`KURT_LMCACHE_LOG_LEVEL`**: LMCache log level (default ``INFO``), forwarded
  as **`LMCACHE_LOG_LEVEL`** so retrieve / hit lines appear in **server.txt**
  without **DEBUG** noise unless you raise it.
- **`KURT_LMCACHE_INTERNAL_API_VERBOSE`**: ``1`` logs failed snapshot URLs.

Examples::

    python3 /app/run_long_doc_qa.py --backend hipfile
    python3 /app/run_long_doc_qa.py --backend native --ready-timeout 7200 \\
        --num-documents 8 --hit-miss-ratio 3:1 --json-output
"""

from __future__ import annotations

import argparse
import json
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
    v = os.environ.get("KURT_CONTAINER_DATA_DIR", "").strip()
    if v:
        return v
    legacy = os.environ.get("VLLM_CONTAINER_DATA_DIR", "").strip()
    return legacy if legacy else "/data"


SERVER_LOG = os.path.join(_data_dir(), "server.txt")
# Dense Llama: avoids MXFP4 MoE on ROCm consumer GPUs. Match vllm-server-* /
# vllm-benchmark (override with env VLLM_MODEL).
_DEFAULT_VLLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _vllm_model() -> str:
    v = os.environ.get("VLLM_MODEL", "").strip()
    return v if v else _DEFAULT_VLLM_MODEL


def _gpu_memory_utilization() -> str:
    v = os.environ.get("KURT_GPU_MEMORY_UTILIZATION", "").strip()
    if not v:
        v = os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "").strip()
    return v if v else "0.72"


def _max_model_len() -> str:
    v = os.environ.get("KURT_MAX_MODEL_LEN", "").strip()
    return v if v else "8192"


def _max_num_batched_tokens() -> str:
    v = os.environ.get("KURT_MAX_NUM_BATCHED_TOKENS", "").strip()
    return v if v else "4096"


def _enforce_eager() -> bool:
    """Avoid TorchInductor autotune peak on tight VRAM; default on."""
    v = os.environ.get("KURT_ENFORCE_EAGER", "").strip().lower()
    if not v:
        v = os.environ.get("VLLM_ENFORCE_EAGER", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return True


def _apply_pytorch_alloc_conf(env: dict[str, str]) -> None:
    v = os.environ.get("KURT_PYTORCH_ALLOC_CONF", "").strip()
    if not v:
        v = os.environ.get("VLLM_PYTORCH_ALLOC_CONF", "").strip()
    if v:
        env["PYTORCH_ALLOC_CONF"] = v


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


def _env_truthy(name: str, default: str = "1") -> bool:
    v = os.environ.get(name, default).strip().lower()
    return v not in ("0", "false", "no", "off", "")


def _chunk_statistics_strategy() -> str:
    v = os.environ.get("KURT_LMCACHE_CHUNK_STATISTICS_STRATEGY", "").strip()
    if v in ("memory_bloom_filter", "file_hash"):
        return v
    return "file_hash"


def _merge_lmcache_extra_config(
    existing_json: str | None, updates: dict[str, object]
) -> str:
    base: dict[str, object] = {}
    if existing_json:
        try:
            parsed = json.loads(existing_json)
            if isinstance(parsed, dict):
                base = parsed
        except json.JSONDecodeError:
            pass
    base.update(updates)
    return json.dumps(base, separators=(",", ":"))


def _chunk_statistics_extra_for_strategy(strategy: str) -> dict[str, object]:
    if strategy == "file_hash":
        out_dir = os.path.join(_data_dir().rstrip("/"), "lmcache_chunk_stats")
        return {
            "chunk_statistics_file_output_dir": out_dir,
            "chunk_statistics_file_rotation_size": 104857600,
            "chunk_statistics_file_max_count": 100,
        }
    return {
        "chunk_statistics_mem_bf_expected_chunks": 20_000_000,
        "chunk_statistics_mem_bf_false_positive_rate": 0.01,
    }


def _apply_chunk_statistics_env(env: dict[str, str]) -> None:
    if not _env_truthy("KURT_LMCACHE_ENABLE_CHUNK_STATISTICS", "1"):
        return
    strategy = _chunk_statistics_strategy()
    env["LMCACHE_ENABLE_CHUNK_STATISTICS"] = "true"
    env["LMCACHE_CHUNK_STATISTICS_AUTO_START_STATISTICS"] = "true"
    env["LMCACHE_CHUNK_STATISTICS_STRATEGY"] = strategy
    extra_updates = _chunk_statistics_extra_for_strategy(strategy)
    if strategy == "file_hash":
        d = extra_updates.get("chunk_statistics_file_output_dir")
        if isinstance(d, str):
            os.makedirs(d, exist_ok=True)
    prior = env.get("LMCACHE_EXTRA_CONFIG")
    env["LMCACHE_EXTRA_CONFIG"] = _merge_lmcache_extra_config(prior, extra_updates)
    env.setdefault("PYTHONHASHSEED", "0")


def _apply_kv_events_env(env: dict[str, str]) -> None:
    if not _env_truthy("KURT_LMCACHE_ENABLE_KV_EVENTS", "1"):
        return
    env["LMCACHE_ENABLE_KV_EVENTS"] = "true"


def _apply_lmcache_log_level(env: dict[str, str]) -> None:
    v = os.environ.get("KURT_LMCACHE_LOG_LEVEL", "").strip()
    env["LMCACHE_LOG_LEVEL"] = v if v else "INFO"


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
        "--max-model-len",
        _max_model_len(),
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        _gpu_memory_utilization(),
        "--block-size",
        "64",
        "--enable-prefix-caching",
        "--stream-interval",
        "20",
        "--max-num-batched-tokens",
        _max_num_batched_tokens(),
        "--async-scheduling",
        "--attention-backend",
        "ROCM_AITER_UNIFIED_ATTN",
        "--kv-transfer-config",
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}',
    ]
    if _enforce_eager():
        out.append("--enforce-eager")
    if load_format_dummy:
        out[3:3] = ["--load-format", "dummy"]
    return out


def _start_server_native(vllm_port: int, lmcache_port: int) -> subprocess.Popen:
    env = os.environ.copy()
    _apply_pytorch_alloc_conf(env)
    env.update(_common_lmcache_env(lmcache_port))
    _apply_chunk_statistics_env(env)
    _apply_kv_events_env(env)
    _apply_lmcache_log_level(env)
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
    _apply_pytorch_alloc_conf(env)
    env.update(_common_lmcache_env(lmcache_port))
    env["LMCACHE_EXTRA_CONFIG"] = _merge_lmcache_extra_config(
        None, {"gds_io_threads": 4}
    )
    _apply_chunk_statistics_env(env)
    _apply_kv_events_env(env)
    _apply_lmcache_log_level(env)
    env.update(
        {
            "HIPFILE_UNSUPPORTED_FILE_SYSTEMS": "true",
            "HIPFILE_ALLOW_COMPAT_MODE": "false",
            "HIPFILE_STATS_LEVEL": "0",
            "LMCACHE_USE_GDS": "true",
            "LMCACHE_GDS_BACKEND": "hipfile",
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


def _collect_lmcache_observability(lmcache_port: int) -> None:
    """Snapshot LMCache internal API (chunk statistics JSON, Prometheus text).

    LMCache binds internal_api_server_port_start + offset per process (e.g.
    scheduler vs worker). We probe a small range so one-GPU ``kv_both`` runs
    still capture available endpoints.
    """
    max_off = int(os.environ.get("KURT_LMCACHE_INTERNAL_API_MAX_OFFSET", "3"))
    root = _data_dir().rstrip("/")
    for off in range(max_off + 1):
        port = lmcache_port + off
        base = f"http://127.0.0.1:{port}"
        for path, ext in (
            ("/chunk_statistics/status", "json"),
            ("/metrics", "txt"),
        ):
            url = base + path
            safe = path.strip("/").replace("/", "_")
            out_path = os.path.join(root, f"lmcache_internal_api_{port}_{safe}.{ext}")
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=20) as resp:
                    raw = resp.read()
                if ext == "json":
                    try:
                        raw = (
                            json.dumps(
                                json.loads(raw.decode("utf-8")),
                                indent=2,
                                sort_keys=True,
                            ).encode("utf-8")
                        )
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
                with open(out_path, "wb") as f:
                    f.write(raw)
                print(f"Wrote LMCache observability snapshot {out_path}", flush=True)
            except (urllib.error.URLError, OSError, TimeoutError, ValueError) as e:
                if _env_truthy("KURT_LMCACHE_INTERNAL_API_VERBOSE", "0"):
                    print(
                        f"LMCache observability skip {url}: {e!r}",
                        flush=True,
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
        if (
            not args.skip_server
            and server is not None
            and server.poll() is None
            and _env_truthy("KURT_LMCACHE_COLLECT_INTERNAL_API", "1")
        ):
            _collect_lmcache_observability(lmcache_port)
        return int(rc)
    finally:
        _terminate_process_group(server)


if __name__ == "__main__":
    raise SystemExit(main())
