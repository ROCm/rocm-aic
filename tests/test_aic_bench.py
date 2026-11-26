# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""aic-bench: the subcommand table, default injection and root resolution."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import aic_bench

BENCH_DIR = REPO_ROOT / "benchmarks"
CONSOLE_SCRIPTS = {"agent-kv", "aiperf"}
BUILTIN = {"list", "env"}
ENDPOINT = "http://vllm-a:8000"


def _no_network(endpoint):
    raise AssertionError(f"model detection attempted against {endpoint}")


def _run(name, model="served", environ=None, detect=_no_network, explicit=()):
    return aic_bench.Run(name, ENDPOINT, "/out", model, frozenset(explicit), stamp="STAMP",
                         environ=environ or {}, detect=detect)


def _build(name, args, **kwargs):
    run = _run(name, **kwargs)
    cmd = aic_bench.build(aic_bench.BY_NAME[name], run, list(args), BENCH_DIR)
    return cmd, run


def _value(cmd, flag):
    return cmd[cmd.index(flag) + 1]


class TableTest(unittest.TestCase):
    def test_subcommand_names(self) -> None:
        self.assertEqual(
            [b.name for b in aic_bench.BENCHES],
            ["list", "env", "cliff", "plot", "cliff-report", "reset-test", "kv-reset-test", "agent-simple",
             "replay-hits", "fixed-load", "serve-sim", "agent-kv", "aiperf"])

    def test_every_subcommand_has_a_target(self) -> None:
        for bench in aic_bench.BENCHES:
            with self.subTest(bench=bench.name):
                self.assertTrue(bench.help)
                if bench.name in BUILTIN:
                    self.assertFalse(bench.script or bench.console)
                elif bench.console:
                    self.assertFalse(bench.script)
                    self.assertIn(bench.console, CONSOLE_SCRIPTS)
                else:
                    self.assertTrue((BENCH_DIR / bench.script).is_file(), bench.script)

    def test_list_names_every_subcommand(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch.object(aic_bench, "detect_model", _no_network):
            self.assertEqual(aic_bench.main(["list"]), 0)
        listed = [line.split()[0] for line in out.getvalue().splitlines()]
        self.assertEqual(listed, [b.name for b in aic_bench.BENCHES])


class SplitTest(unittest.TestCase):
    def test_global_options_end_at_the_subcommand(self) -> None:
        self.assertEqual(
            aic_bench.split_argv(["--endpoint", "http://x:1", "--out=/o", "cliff", "--model", "m"]),
            (["--endpoint", "http://x:1", "--out=/o"], ["cliff", "--model", "m"]))
        self.assertEqual(aic_bench.split_argv(["agent-kv", "run", "--help"]), ([], ["agent-kv", "run", "--help"]))
        self.assertEqual(aic_bench.split_argv(["--help"]), (["--help"], []))


class RootTest(unittest.TestCase):
    def test_resolution_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            live, here = Path(tmp) / "live", Path(tmp) / "here"
            here.mkdir()
            self.assertEqual(aic_bench.bench_root({}, live, here), here)
            live.mkdir()
            self.assertEqual(aic_bench.bench_root({}, live, here), here)
            (live / "aic_bench.py").touch()
            self.assertEqual(aic_bench.bench_root({}, live, here), live)
            self.assertEqual(aic_bench.bench_root({"AIC_BENCH_ROOT": "/else"}, live, here), Path("/else"))

    def test_defaults(self) -> None:
        self.assertEqual(aic_bench.LIVE_ROOT, Path("/repo/benchmarks"))
        if not (aic_bench.LIVE_ROOT / "aic_bench.py").is_file():
            self.assertEqual(aic_bench.bench_root({}), BENCH_DIR)


class CliffTest(unittest.TestCase):
    def test_defaults_are_filled(self) -> None:
        cmd, run = _build("cliff", ["--arm", "kvd_v2"])
        self.assertEqual(cmd[:3], [sys.executable, "-u", str(BENCH_DIR / "run_cliff.py")])
        self.assertEqual(_value(cmd, "--endpoint"), ENDPOINT)
        self.assertEqual(_value(cmd, "--model"), "served")
        self.assertEqual(_value(cmd, "--out"), "/out/cliff/STAMP/kvd_v2.csv")
        self.assertEqual(run.mkdirs, ["/out/cliff/STAMP"])

    def test_user_flags_win(self) -> None:
        for given in (["--endpoint", "http://mine:1", "--model", "mine", "--out", "/mine.csv"],
                      ["--endpoint=http://mine:1", "--model=mine", "--out=/mine.csv"]):
            with self.subTest(given=given):
                cmd, run = _build("cliff", ["--arm", "kvd_v2", *given], model=None)
                self.assertEqual(cmd[3:], ["--arm", "kvd_v2", *given])
                self.assertEqual(run.mkdirs, [])

    def test_abbreviated_user_flags_win(self) -> None:
        given = ["--arm", "kvd_v2", "--endp", "http://mine:1", "--mod=mine", "--ou", "/mine.csv"]
        cmd, run = _build("cliff", given, model=None)
        self.assertEqual(cmd[3:], given)
        self.assertEqual(run.mkdirs, [])

    def test_model_is_detected_only_when_missing(self) -> None:
        cmd, run = _build("cliff", [], model=None, detect=lambda endpoint: f"first@{endpoint}")
        self.assertEqual(_value(cmd, "--model"), f"first@{ENDPOINT}")
        self.assertEqual(run.env["AIC_BENCH_MODEL"], f"first@{ENDPOINT}")

    def test_help_is_passed_through_untouched(self) -> None:
        cmd, run = _build("cliff", ["--help"], model=None)
        self.assertEqual(cmd[3:], ["--help"])
        self.assertEqual(run.mkdirs, [])


class ResetTestTest(unittest.TestCase):
    def test_environment_is_filled(self) -> None:
        cmd, run = _build("reset-test", [])
        self.assertEqual(cmd[3:], [])
        self.assertEqual(run.env["AIC_TEST_VLLM_URL"], ENDPOINT)
        self.assertEqual(run.env["AIC_TEST_LMCACHE_URL"], "http://vllm-a:8080")
        self.assertEqual(run.env["AIC_TEST_MODEL"], "served")
        self.assertEqual(run.env["AIC_BENCH_ENDPOINT"], ENDPOINT)

    def test_caller_environment_wins(self) -> None:
        mine = {"AIC_TEST_VLLM_URL": "http://mine:1/", "AIC_TEST_LMCACHE_URL": "http://mine:2",
                "AIC_TEST_MODEL": "mine"}
        _, run = _build("reset-test", [], model=None, environ=mine)
        self.assertEqual(run.env["AIC_TEST_VLLM_URL"], "http://mine:1")
        self.assertEqual(run.env["AIC_BENCH_ENDPOINT"], "http://mine:1")
        for name in ("AIC_TEST_LMCACHE_URL", "AIC_TEST_MODEL"):
            self.assertNotIn(name, run.env)

    def test_peers_follow_the_caller_url(self) -> None:
        _, run = _build("reset-test", [], model=None, environ={"AIC_TEST_VLLM_URL": "http://mine:1"},
                        detect=lambda endpoint: f"first@{endpoint}")
        self.assertEqual(run.env["AIC_TEST_LMCACHE_URL"], "http://mine:8080")
        self.assertEqual(run.env["AIC_TEST_MODEL"], "first@http://mine:1")

    def test_command_line_beats_the_caller_environment(self) -> None:
        stale = {"AIC_TEST_VLLM_URL": "http://stale:8000", "AIC_TEST_MODEL": "stale"}
        _, run = _build("reset-test", [], environ=stale, explicit=("endpoint", "model"))
        self.assertEqual(run.env["AIC_TEST_VLLM_URL"], ENDPOINT)
        self.assertEqual(run.env["AIC_TEST_LMCACHE_URL"], "http://vllm-a:8080")
        self.assertEqual(run.env["AIC_TEST_MODEL"], "served")

    def test_lmcache_http_override(self) -> None:
        _, run = _build("reset-test", [], environ={"AIC_BENCH_LMCACHE_HTTP_URL": "http://lm:9"})
        self.assertEqual(run.env["AIC_TEST_LMCACHE_URL"], "http://lm:9")


class AgentKvTest(unittest.TestCase):
    def test_run_defaults(self) -> None:
        cmd, run = _build("agent-kv", ["run", "--num-books", "2"])
        self.assertEqual(cmd[:4], ["agent-kv", "run", "--num-books", "2"])
        self.assertEqual(_value(cmd, "--base-url"), ENDPOINT + "/v1")
        self.assertEqual(_value(cmd, "--model"), "served")
        self.assertEqual(_value(cmd, "--cache-dir"), "/out/agent-kv/books")
        self.assertEqual(_value(cmd, "--output"), "/out/agent-kv/STAMP/agent-kv-results.json")
        self.assertNotIn("--lmcache-http-url", cmd)
        self.assertEqual(run.mkdirs, ["/out/agent-kv/STAMP"])

    def test_run_user_flags_win(self) -> None:
        for given in (["--base-url", "http://mine/v1", "--model", "mine", "--cache-dir", "/c", "--output", "/o.json"],
                      ["--base-url=http://mine/v1", "--model=mine", "--cache-dir=/c", "--output=/o.json"],
                      ["--base-url=http://mine/v1", "--model=mine", "--cache-dir=/c", "-o", "/o.json"],
                      ["--base-url=http://mine/v1", "--model=mine", "--cache-dir=/c", "-o/o.json"]):
            with self.subTest(given=given):
                cmd, run = _build("agent-kv", ["run", *given], model=None)
                self.assertEqual(cmd, ["agent-kv", "run", *given])
                self.assertEqual(run.mkdirs, [])

    def test_host_or_port_suppresses_base_url(self) -> None:
        for given in (["--host", "h"], ["--port=9"]):
            with self.subTest(given=given):
                cmd, _ = _build("agent-kv", ["run", *given])
                self.assertNotIn("--base-url", cmd)

    def test_nvme_sweep_defaults(self) -> None:
        cmd, _ = _build("agent-kv", ["nvme-sweep"])
        self.assertEqual(_value(cmd, "--base-url"), ENDPOINT + "/v1")
        self.assertEqual(_value(cmd, "--lmcache-http-url"), "http://vllm-a:8080")
        self.assertEqual(_value(cmd, "--output"), "/out/agent-kv/STAMP/nvme-sweep-results.json")

    def test_nvme_sweep_user_flags_win(self) -> None:
        for given in (["--lmcache-http-url", "http://mine:1", "--model", "mine"],
                      ["--lmcache-http-url=http://mine:1", "--model=mine"]):
            with self.subTest(given=given):
                cmd, _ = _build("agent-kv", ["nvme-sweep", *given], model=None)
                self.assertEqual(cmd[:2 + len(given)], ["agent-kv", "nvme-sweep", *given])
                for flag in ("--lmcache-http-url", "--model"):
                    self.assertEqual(sum(arg.startswith(flag) for arg in cmd), 1, flag)

    def test_other_subcommands_get_nothing(self) -> None:
        for args in ([], ["--version"], ["man"], ["run", "--help"]):
            with self.subTest(args=args):
                cmd, run = _build("agent-kv", args, model=None)
                self.assertEqual(cmd, ["agent-kv", *args])
                self.assertEqual(run.mkdirs, [])


class PassThroughTest(unittest.TestCase):
    def test_aiperf_gets_no_injection(self) -> None:
        cmd, run = _build("aiperf", ["profile", "--url", "http://x"], model=None)
        self.assertEqual(cmd, ["aiperf", "profile", "--url", "http://x"])
        self.assertEqual(run.mkdirs, [])

    def test_kv_drivers_get_the_endpoint_by_environment(self) -> None:
        for name in ("agent-simple", "replay-hits", "fixed-load"):
            with self.subTest(name=name):
                cmd, run = _build(name, ["r1", "4"], model=None)
                self.assertEqual(cmd[3:], ["r1", "4"])
                self.assertEqual(run.env, {"AIC_BENCH_ENDPOINT": ENDPOINT, "AIC_BENCH_OUT": "/out"})

    def test_output_directories(self) -> None:
        cmd, _ = _build("plot", ["--input", "/in"])
        self.assertEqual(_value(cmd, "--output-dir"), "/out/plot/STAMP")
        cmd, _ = _build("serve-sim", ["--output=/mine"])
        self.assertNotIn("--output", cmd)
        self.assertEqual(_value(cmd, "--base-url"), ENDPOINT)
        for name, flag in (("serve-sim", "--output"), ("plot", "--output-dir"), ("cliff-report", "--output-dir")):
            with self.subTest(name=name):
                cmd, run = _build(name, ["--out", "/mine"])
                self.assertNotIn(flag, cmd)
                self.assertEqual(run.mkdirs, [])


class EndpointTest(unittest.TestCase):
    def test_peer_url(self) -> None:
        self.assertEqual(aic_bench.peer_url("http://vllm-a:8000", 8080), "http://vllm-a:8080")
        self.assertEqual(aic_bench.peer_url("http://[::1]:8000", 8080), "http://[::1]:8080")

    def test_endpoint_must_be_a_url(self) -> None:
        self.assertEqual(aic_bench.checked_endpoint("http://e:8000/"), "http://e:8000")
        for bad in ("aic-lmcache:8000", "aic-lmcache", "http://"):
            with self.subTest(bad=bad), self.assertRaises(SystemExit) as caught:
                aic_bench.checked_endpoint(bad)
            self.assertIn(repr(bad), str(caught.exception))


class MainTest(unittest.TestCase):
    def test_main_execs_the_tool_with_the_merged_environment(self) -> None:
        with tempfile.TemporaryDirectory() as out, \
                mock.patch.object(aic_bench, "execute") as execute, \
                mock.patch.object(aic_bench, "detect_model", _no_network):
            aic_bench.main(["--endpoint", "http://e:8000/", "--model", "m", f"--out={out}",
                            "cliff", "--arm", "vram_only"])
            cmd, env = execute.call_args.args
            self.assertEqual(_value(cmd, "--endpoint"), "http://e:8000")
            self.assertEqual(env["AIC_BENCH_MODEL"], "m")
            self.assertTrue(Path(_value(cmd, "--out")).parent.is_dir())

    def test_command_line_options_are_recorded(self) -> None:
        with mock.patch.object(aic_bench, "execute"), \
                mock.patch.object(aic_bench, "build", return_value=["true"]) as build:
            aic_bench.main(["--endpoint=http://e:8000", "--model", "m", "reset-test"])
            self.assertEqual(build.call_args.args[1].explicit, {"endpoint", "model"})

    def test_env_reports_the_served_model(self) -> None:
        for detect, expected in ((mock.Mock(return_value="first"), "model     first"),
                                 (mock.Mock(side_effect=OSError("refused")), "refused")):
            out = io.StringIO()
            with self.subTest(expected=expected), contextlib.redirect_stdout(out), \
                    mock.patch.object(aic_bench, "first_model", detect):
                self.assertEqual(aic_bench.main(["--endpoint", "http://e:8000", "env"]), 0)
                self.assertIn("endpoint  http://e:8000", out.getvalue())
                self.assertIn(expected, out.getvalue())

    def test_unreachable_endpoint_is_a_clear_error(self) -> None:
        with mock.patch("urllib.request.urlopen", side_effect=OSError("refused")), \
                self.assertRaises(SystemExit) as caught:
            aic_bench.detect_model("http://nowhere:8000")
        self.assertIn("http://nowhere:8000/v1/models", str(caught.exception))
        self.assertIn("--model", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
