# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Regression tests for the cliff benchmark's request-failure status."""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import run_cliff


class _HealthyResponse:
    status_code = 200
    text = ""

    def json(self):
        return {}


class _AsyncClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        pass

    async def get(self, *args, **kwargs) -> _HealthyResponse:
        return _HealthyResponse()


class _JsonResponse:
    def __init__(self, body: dict, status_code: int = 200, text: str = "") -> None:
        self._body = body
        self.status_code = status_code
        self.text = text

    def json(self) -> dict:
        return self._body


class _RecordingAsyncClient:
    def __init__(self, response: _JsonResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, dict, float]] = []

    async def post(self, url: str, *, json: dict, timeout: float) -> _JsonResponse:
        self.calls.append((url, json, timeout))
        return self.response


async def _no_calibration(*args, **kwargs) -> None:
    pass


def _result(error: str | None) -> run_cliff.ReqResult:
    return run_cliff.ReqResult(
        client_id=0,
        run_id=0,
        issued_at=1.0,
        finished_at=2.0,
        prompt_chars=40,
        prompt_tokens_reported=10,
        output_tokens=1,
        error=error,
    )


class RequestFailureStatusTest(unittest.TestCase):
    def _run(
        self,
        *,
        error: str | None,
        allow_errors: bool,
        warmup_error: str | None = None,
        concurrency: str = "1",
        arm: str = "vram_only",
    ) -> tuple[int, dict[str, str], list[str]]:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "cliff.csv"
            args = argparse.Namespace(
                endpoint="http://test.invalid:8000",
                model="test-model",
                tokenizer=None,
                arm=arm,
                isl=2,
                shared_prefix_tokens=1,
                max_tokens=1,
                concurrencies=concurrency,
                iters=1,
                warmup_iters=0,
                warmup_at_each_c=warmup_error is not None,
                post_warmup_sleep_s=0.0,
                prefix_mode="shared",
                request_timeout=1.0,
                allow_request_errors=allow_errors,
                metrics_endpoint=None,
                no_metrics=True,
                metrics_settle_interval=0.01,
                metrics_settle_timeout=0.0,
                out=str(output),
                append=False,
            )
            fake_httpx = types.SimpleNamespace(AsyncClient=_AsyncClient)
            waves = [(1.0, [_result(error)])]
            if warmup_error is not None:
                waves.insert(0, (1.0, [_result(warmup_error)]))
            wave = mock.AsyncMock(side_effect=waves)
            with (
                mock.patch.dict(sys.modules, {"httpx": fake_httpx}),
                mock.patch.object(run_cliff, "set_active_tokenizer"),
                mock.patch.object(run_cliff, "calibrate_word_ratio", _no_calibration),
                mock.patch.object(run_cliff, "_run_one_concurrency", wave),
                mock.patch.object(run_cliff, "_ckpt") as checkpoint,
            ):
                status = asyncio.run(run_cliff.amain(args))
            with output.open(newline="") as handle:
                row = next(csv.DictReader(handle))
        messages = [call.args[0] for call in checkpoint.call_args_list]
        return status, row, messages

    def test_successful_requests_return_success(self) -> None:
        status, row, _ = self._run(error=None, allow_errors=False)

        self.assertEqual(status, 0)
        self.assertEqual(row["ok_count"], "1")
        self.assertEqual(row["err_count"], "0")

    def test_request_errors_fail_after_writing_diagnostics(self) -> None:
        status, row, _ = self._run(error="http 500: broken", allow_errors=False)

        self.assertEqual(status, 1)
        self.assertEqual(row["ok_count"], "0")
        self.assertEqual(row["err_count"], "1")

    def test_request_errors_can_be_allowed_explicitly(self) -> None:
        status, row, _ = self._run(error="ReadTimeout: timed out", allow_errors=True)

        self.assertEqual(status, 0)
        self.assertEqual(row["ok_count"], "0")
        self.assertEqual(row["err_count"], "1")

    def test_per_concurrency_warmup_logs_first_error(self) -> None:
        status, row, messages = self._run(
            error=None,
            allow_errors=False,
            warmup_error="http 500: warmup broken",
            concurrency="3",
        )

        self.assertEqual(status, 1)
        self.assertEqual(row["err_count"], "0")
        self.assertIn(
            "  c=3 per-c warmup first error: http 500: warmup broken", messages
        )

    def test_main_propagates_benchmark_status(self) -> None:
        async def fail(_args: argparse.Namespace) -> int:
            return 1

        argv = [
            "run_cliff.py",
            "--endpoint", "http://test.invalid:8000",
            "--model", "test-model",
            "--arm", "vram_only",
            "--out", "unused.csv",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(run_cliff, "amain", fail),
            self.assertRaises(SystemExit) as raised,
        ):
            run_cliff.main()

        self.assertEqual(raised.exception.code, 1)

    def test_kvbench_fire_one_uses_chat_completions_api(self) -> None:
        client = _RecordingAsyncClient(
            _JsonResponse({
                "usage": {"prompt_tokens": 12, "completion_tokens": 1},
            })
        )

        result = asyncio.run(
            run_cliff._fire_one(
                client,
                "http://test.invalid:8000",
                "llama-3.1-8b",
                "prompt text",
                client_id=0,
                run_id=0,
                max_tokens=1,
                request_timeout=5.0,
                arm="kvbench",
            )
        )

        self.assertIsNone(result.error)
        self.assertEqual(result.prompt_tokens_reported, 12)
        self.assertEqual(result.output_tokens, 1)
        self.assertEqual(client.calls[0][0], "http://test.invalid:8000/v1/chat/completions")
        self.assertEqual(
            client.calls[0][1]["messages"],
            [{"role": "user", "content": "prompt text"}],
        )
        self.assertNotIn("prompt", client.calls[0][1])

    def test_kvbench_arm_skips_vllm_metrics_scrape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "cliff.csv"
            args = argparse.Namespace(
                endpoint="http://test.invalid:8000",
                model="test-model",
                tokenizer=None,
                arm="kvbench",
                isl=2,
                shared_prefix_tokens=1,
                max_tokens=1,
                concurrencies="1",
                iters=1,
                warmup_iters=0,
                warmup_at_each_c=False,
                post_warmup_sleep_s=0.0,
                prefix_mode="shared",
                request_timeout=1.0,
                allow_request_errors=False,
                metrics_endpoint=None,
                no_metrics=False,
                metrics_settle_interval=0.01,
                metrics_settle_timeout=0.0,
                out=str(output),
                append=False,
            )
            fake_httpx = types.SimpleNamespace(AsyncClient=_AsyncClient)
            wave = mock.AsyncMock(return_value=(1.0, [_result(None)]))
            snap = mock.AsyncMock(side_effect=AssertionError("metrics scrape should be skipped"))
            with (
                mock.patch.dict(sys.modules, {"httpx": fake_httpx}),
                mock.patch.object(run_cliff, "set_active_tokenizer") as set_tokenizer,
                mock.patch.object(run_cliff, "calibrate_word_ratio", _no_calibration),
                mock.patch.object(run_cliff, "_run_one_concurrency", wave),
                mock.patch.object(run_cliff, "_snap_cache_settled", snap),
            ):
                status = asyncio.run(run_cliff.amain(args))
            with output.open(newline="") as handle:
                row = next(csv.DictReader(handle))

        self.assertEqual(status, 0)
        self.assertEqual(row["arm"], "kvbench")
        self.assertEqual(row["l1_cache_queries"], "")
        self.assertEqual(row["ext_cache_queries"], "")
        snap.assert_not_called()
        set_tokenizer.assert_called_once_with(None)


if __name__ == "__main__":
    unittest.main()
