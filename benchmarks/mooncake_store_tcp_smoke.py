#!/usr/bin/env python3
# Copyright (c) 2026 Ziming Qi
#
# SPDX-License-Identifier: MIT

"""Check cross-client Mooncake Store semantics over TCP.

This probe expects a running ``mooncake_master`` and a Python environment with
Mooncake Store installed. It deliberately uses separate writer and reader
processes so a pass cannot come from one client's local state.

Example:

    python3 benchmarks/mooncake_store_tcp_smoke.py \
        --master 127.0.0.1:50051 \
        --key-prefix aic-mooncake-$(date +%s)

The probe covers aligned and unaligned payloads, a missing key, and deletion.
It does not exercise RDMA, GPU buffers, LMCache, or vLLM.

``--lease-wait`` must exceed the master's KV lease TTL so removal can succeed.
Its default includes a 0.5-second margin over Mooncake's 10-second default;
override it when the master uses a different TTL.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback


DEFAULT_MASTER_LEASE_TTL_S = 10.0
DEFAULT_LEASE_WAIT_S = DEFAULT_MASTER_LEASE_TTL_S + 0.5


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def payload(size: int, salt: int) -> bytes:
    return bytes((((index * 31) + salt) % 251) + 1 for index in range(size))


def setup_store(local_hostname: str, master: str):
    try:
        from mooncake.store import MooncakeDistributedStore
    except ImportError as error:
        raise RuntimeError("Mooncake Store is not installed") from error

    store = MooncakeDistributedStore()
    result = store.setup(
        local_hostname,
        "P2PHANDSHAKE",
        64 * 1024 * 1024,
        16 * 1024 * 1024,
        "tcp",
        "",
        master,
    )
    require(result == 0, f"store setup failed for {local_hostname}: {result}")
    return store


def writer(
    connection,
    local_hostname: str,
    master: str,
    key_prefix: str,
    lease_wait: float,
) -> None:
    store = None
    try:
        store = setup_store(local_hostname, master)
        entries = {
            f"{key_prefix}-aligned": payload(4096, 17),
            f"{key_prefix}-unaligned": payload(4099, 29),
        }
        results = {key: store.put(key, value) for key, value in entries.items()}
        require(all(result == 0 for result in results.values()), f"put failed: {results}")
        connection.send({"stage": "ready", "results": results})

        require(connection.recv() == "delete", "writer received unexpected command")
        time.sleep(lease_wait)
        results = {key: store.remove(key) for key in entries}
        require(
            all(result == 0 for result in results.values()),
            f"remove failed: {results}",
        )
        connection.send({"stage": "deleted", "results": results})
        require(connection.recv() == "close", "writer received unexpected close")
    except Exception as error:
        connection.send(
            {
                "stage": "error",
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        if store is not None:
            store.close()
        connection.close()


def reader(connection, local_hostname: str, master: str, key_prefix: str) -> None:
    store = None
    try:
        store = setup_store(local_hostname, master)
        expected = {
            f"{key_prefix}-aligned": payload(4096, 17),
            f"{key_prefix}-unaligned": payload(4099, 29),
        }
        actual = {key: store.get(key) for key in expected}
        require(actual == expected, "cross-client payload mismatch")

        missing = store.get(f"{key_prefix}-missing")
        require(missing in (b"", None), "missing key unexpectedly returned data")
        connection.send(
            {
                "stage": "read",
                "lengths": {key: len(value) for key, value in actual.items()},
            }
        )

        require(
            connection.recv() == "check-delete",
            "reader received unexpected command",
        )
        after_delete = {key: store.get(key) for key in expected}
        require(
            all(value in (b"", None) for value in after_delete.values()),
            "removed key is still readable",
        )
        connection.send({"stage": "post-delete", "misses": len(after_delete)})
        require(connection.recv() == "close", "reader received unexpected close")
    except Exception as error:
        connection.send(
            {
                "stage": "error",
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        if store is not None:
            store.close()
        connection.close()


def receive(connection, label: str, timeout: float) -> dict:
    require(connection.poll(timeout), f"timed out waiting for {label}")
    message = connection.recv()
    require(message.get("stage") != "error", f"{label} failed: {message}")
    return message


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", default="127.0.0.1:50051")
    parser.add_argument("--writer", default="127.0.0.1:50071")
    parser.add_argument("--reader", default="127.0.0.1:50072")
    parser.add_argument("--key-prefix", required=True)
    parser.add_argument(
        "--lease-wait",
        type=float,
        default=DEFAULT_LEASE_WAIT_S,
        help="seconds to wait before removal; must exceed the master lease TTL "
        f"(default: {DEFAULT_LEASE_WAIT_S:g}s for Mooncake's 10s default TTL)",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    require(args.writer != args.reader, "writer and reader endpoints must differ")
    require(args.lease_wait >= 0, "lease wait must be non-negative")
    require(args.timeout > args.lease_wait, "timeout must exceed lease wait")

    context = mp.get_context("spawn")
    parent_writer, child_writer = context.Pipe()
    parent_reader, child_reader = context.Pipe()
    writer_process = context.Process(
        target=writer,
        args=(child_writer, args.writer, args.master, args.key_prefix, args.lease_wait),
        name="mooncake-writer",
    )
    reader_process = context.Process(
        target=reader,
        args=(child_reader, args.reader, args.master, args.key_prefix),
        name="mooncake-reader",
    )

    writer_process.start()
    try:
        writer_result = receive(parent_writer, "writer ready", args.timeout)
        require(writer_result["stage"] == "ready", f"unexpected result: {writer_result}")

        reader_process.start()
        reader_result = receive(parent_reader, "reader fetch", args.timeout)
        require(reader_result["stage"] == "read", f"unexpected result: {reader_result}")

        parent_writer.send("delete")
        delete_result = receive(parent_writer, "writer delete", args.timeout)
        require(delete_result["stage"] == "deleted", f"unexpected result: {delete_result}")

        parent_reader.send("check-delete")
        post_delete = receive(parent_reader, "reader delete check", args.timeout)
        require(post_delete["stage"] == "post-delete", f"unexpected result: {post_delete}")

        parent_writer.send("close")
        parent_reader.send("close")
        writer_process.join(args.timeout)
        reader_process.join(args.timeout)
        require(writer_process.exitcode == 0, f"writer exit code: {writer_process.exitcode}")
        require(reader_process.exitcode == 0, f"reader exit code: {reader_process.exitcode}")

        print(
            json.dumps(
                {
                    "key_prefix": args.key_prefix,
                    "master": args.master,
                    "protocol": "tcp",
                    "reader": args.reader,
                    "result": "PASS",
                    "sizes": sorted(reader_result["lengths"].values()),
                    "writer": args.writer,
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        for process in (writer_process, reader_process):
            if process.is_alive():
                process.terminate()
                process.join(5)
        parent_writer.close()
        parent_reader.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"result": "FAIL", "error": repr(error)}, sort_keys=True))
        raise
