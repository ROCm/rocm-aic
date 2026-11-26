#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""aic-bench: one entry point for every benchmark client in the aic-client image.

    aic-bench [--endpoint URL] [--model ID] [--out DIR] <subcommand> [args...]

Everything after the subcommand goes to the underlying tool unchanged;
`aic-bench <subcommand> --help` shows that tool's help.  aic-bench only fills in
the endpoint, the model and an output location the tool was not given, using
the tool's own flag or environment variable.

    --endpoint / AIC_BENCH_ENDPOINT   server base URL, no /v1 (default http://aic-lmcache:8000)
    --model    / AIC_BENCH_MODEL      served model (default: the first one the endpoint lists)
    --out      / AIC_BENCH_OUT        results root (default /logs/bench); a run writes to
                                      <out>/<subcommand>/<UTC timestamp>/

The LMCache HTTP API is taken to be port 8080 on the endpoint's host unless
AIC_BENCH_LMCACHE_HTTP_URL names it.  Benchmark scripts are run from
$AIC_BENCH_ROOT, else /repo/benchmarks when it holds aic_bench.py (a live
checkout needs no image rebuild), else the directory holding this file.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

DEFAULT_ENDPOINT = "http://aic-lmcache:8000"
DEFAULT_OUT = "/logs/bench"
LIVE_ROOT = Path("/repo/benchmarks")
GLOBAL_FLAGS = ("--endpoint", "--model", "--out")
AGENT_KV_OUTPUTS = {
    "run": "agent-kv-results.json",
    "run-ramp": "agent-kv-ramp-results.json",
    "nvme-sweep": "nvme-sweep-results.json",
}


def bench_root(env=os.environ, live: Path = LIVE_ROOT, here: Path = Path(__file__).resolve().parent) -> Path:
    if env.get("AIC_BENCH_ROOT"):
        return Path(env["AIC_BENCH_ROOT"])
    return live if (live / "aic_bench.py").is_file() else here


def first_model(endpoint: str) -> str:
    with urllib.request.urlopen(endpoint + "/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def detect_model(endpoint: str) -> str:
    try:
        return first_model(endpoint)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"aic-bench: cannot read the served model from {endpoint}/v1/models ({e}); "
                 "check --endpoint or pass --model")


def checked_endpoint(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        sys.exit(f"aic-bench: endpoint {url!r} is not an http(s) URL such as {DEFAULT_ENDPOINT}")
    return url.rstrip("/")


def peer_url(endpoint: str, port: int) -> str:
    """<port> on the endpoint's host."""
    host = urlsplit(endpoint).hostname
    return f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"


@dataclass
class Run:
    """The settings of one invocation and what it adds to the tool's environment."""
    name: str
    endpoint: str
    out: str
    model_id: str | None = None
    explicit: frozenset = frozenset()   # endpoint, model, out: those given on the command line
    stamp: str = field(default_factory=lambda: time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    environ: dict = field(default_factory=lambda: dict(os.environ))
    detect: Callable[[str], str] = detect_model
    env: dict = field(default_factory=dict)
    mkdirs: list = field(default_factory=list)

    def model(self) -> str:
        if not self.model_id:
            self.model_id = self.detect(self.endpoint)
        return self.model_id

    def run_dir(self) -> str:
        path = f"{self.out}/{self.name}/{self.stamp}"
        if path not in self.mkdirs:
            self.mkdirs.append(path)
        return path

    def lmcache_http(self) -> str:
        return self.environ.get("AIC_BENCH_LMCACHE_HTTP_URL") or peer_url(self.endpoint, 8080)

    def setenv(self, name: str, value) -> None:
        """Set <name> for the tool unless the caller's environment already has it."""
        if not self.environ.get(name):
            self.env[name] = value() if callable(value) else value


def given(args: list[str], *flags: str, abbrev: bool = True) -> bool:
    """True if <args> carries one of <flags>, as `--flag value`, `--flag=value` or `-fvalue`.

    With <abbrev>, a prefix of a long flag counts too: argparse, which every
    script here uses, accepts `--out` for `--output`.
    """
    for arg in args:
        name = arg.split("=", 1)[0]
        for flag in flags:
            if not flag.startswith("--"):
                if arg.startswith(flag) and not arg.startswith("--"):
                    return True
            elif name == flag or (abbrev and len(name) > 2 and flag.startswith(name)):
                return True
    return False


def fill(args: list[str], flag: str, value, *aliases: str, abbrev: bool = True) -> list[str]:
    """Append `flag value` unless the user passed the flag (or an alias)."""
    if given(args, flag, *aliases, abbrev=abbrev):
        return args
    return args + [flag, value() if callable(value) else value]


def flag_value(args: list[str], flag: str) -> str | None:
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def _cliff(run: Run, args: list[str]) -> list[str]:
    args = fill(args, "--endpoint", run.endpoint)
    args = fill(args, "--model", run.model)
    return fill(args, "--out", lambda: f"{run.run_dir()}/{flag_value(args, '--arm') or 'cliff'}.csv")


def _output_dir(run: Run, args: list[str]) -> list[str]:
    return fill(args, "--output-dir", run.run_dir)


def _reset_test(run: Run, args: list[str]) -> list[str]:
    ambient = run.environ.get("AIC_TEST_VLLM_URL")
    if ambient and "endpoint" not in run.explicit:
        run.endpoint = checked_endpoint(ambient)
    run.env["AIC_TEST_VLLM_URL"] = run.endpoint
    run.setenv("AIC_TEST_LMCACHE_URL", run.lmcache_http)
    if "model" in run.explicit:
        run.env["AIC_TEST_MODEL"] = run.model_id
    else:
        run.setenv("AIC_TEST_MODEL", run.model)
    return args


def _kv_reset_test(run: Run, args: list[str]) -> list[str]:
    ambient = run.environ.get("AIC_TEST_VLLM_URL")
    if ambient and "endpoint" not in run.explicit:
        run.endpoint = checked_endpoint(ambient)
    run.env["AIC_TEST_VLLM_URL"] = run.endpoint
    if "model" in run.explicit:
        run.env["AIC_TEST_MODEL"] = run.model_id
    else:
        run.setenv("AIC_TEST_MODEL", run.model)
    return args


def _serve_sim(run: Run, args: list[str]) -> list[str]:
    args = fill(args, "--base-url", run.endpoint)
    args = fill(args, "--model", run.model)
    return fill(args, "--output", run.run_dir)


def _agent_kv(run: Run, args: list[str]) -> list[str]:
    sub = next((a for a in args if not a.startswith("-")), None)
    if sub not in AGENT_KV_OUTPUTS:
        return args
    exact = functools.partial(fill, abbrev=False)   # click takes no abbreviations
    if not given(args, "--host", "--port", abbrev=False):
        args = exact(args, "--base-url", run.endpoint + "/v1")
    args = exact(args, "--model", run.model)
    args = exact(args, "--cache-dir", f"{run.out}/agent-kv/books")
    args = exact(args, "--output", lambda: f"{run.run_dir()}/{AGENT_KV_OUTPUTS[sub]}", "-o")
    if sub == "nvme-sweep":
        args = exact(args, "--lmcache-http-url", run.lmcache_http)
    return args


@dataclass(frozen=True)
class Bench:
    name: str
    help: str
    script: str = ""    # path under the benchmark root, run with this interpreter
    console: str = ""   # console script on PATH
    prepare: Callable[[Run, list[str]], list[str]] | None = None


BENCHES = (
    Bench("list", "List the subcommands"),
    Bench("env", "Print the resolved endpoint, model, output root and benchmark root"),
    Bench("cliff", "Concurrency sweep with a shared or per-client prefix (run_cliff.py)",
          script="run_cliff.py", prepare=_cliff),
    Bench("plot", "PNG charts from cliff CSVs (plot_cliff.py)",
          script="plot_cliff.py", prepare=_output_dir),
    Bench("cliff-report", "HTML page from the run directories of the cliff Slurm jobs (cliff_to_html.py)",
          script="cliff_to_html.py", prepare=_output_dir),
    Bench("reset-test", "L1 and L2 retrieval test using /reset_prefix_cache (vllm_reset_test.py)",
          script="vllm_reset_test.py", prepare=_reset_test),
    Bench("kv-reset-test", "External KV cache retrieval after /reset_prefix_cache, any connector (kv_reset_test.py)",
          script="kv_reset_test.py", prepare=_kv_reset_test),
    Bench("agent-simple", "Synthetic mixed-size cache-tier stress (agent-simple/stress.py)",
          script="agent-simple/stress.py"),
    Bench("replay-hits", "Evict prompts from the GPU cache, replay them for L1/L2 hits (common/replay_hits.py)",
          script="common/replay_hits.py"),
    Bench("fixed-load", "Fixed workload for comparing hsa-snoop builds (common/fixed_load.py)",
          script="common/fixed_load.py"),
    Bench("serve-sim", "Multi-turn ShareGPT sessions with think time (aic_bench_serve.py)",
          script="aic_bench_serve.py", prepare=_serve_sim),
    Bench("agent-kv", "KV-reuse ramp over Gutenberg books: agent-kv run | run-ramp | nvme-sweep",
          console="agent-kv", prepare=_agent_kv),
    Bench("aiperf", "aiperf, the AgentX client; arguments are passed as given",
          console="aiperf"),
)
BY_NAME = {b.name: b for b in BENCHES}


def listing() -> str:
    width = max(len(b.name) for b in BENCHES)
    return "\n".join(f"  {b.name:<{width}}  {b.help}" for b in BENCHES)


def split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split into aic-bench's own options and `<subcommand> [args...]`."""
    i = 0
    while i < len(argv) and argv[i].startswith("-"):
        i += 2 if argv[i] in GLOBAL_FLAGS else 1
    return argv[:i], argv[i:]


def build(bench: Bench, run: Run, args: list[str], root: Path) -> list[str]:
    """The tool's argv; fills run.env and run.mkdirs as a side effect."""
    if bench.prepare and not (given(args, "--help") or "-h" in args):
        args = bench.prepare(run, args)
    run.env.update(AIC_BENCH_ENDPOINT=run.endpoint, AIC_BENCH_OUT=run.out)
    if run.model_id:
        run.env["AIC_BENCH_MODEL"] = run.model_id
    if bench.script:
        return [sys.executable, "-u", str(root / bench.script), *args]
    return [bench.console, *args]


def execute(cmd: list[str], env: dict) -> None:
    sys.stdout.flush()
    try:
        os.execvpe(cmd[0], cmd, env)
    except OSError as e:
        sys.exit(f"aic-bench: cannot run {cmd[0]}: {e}")


def main(argv: list[str] | None = None) -> int:
    own, rest = split_argv(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        prog="aic-bench", usage="aic-bench [--endpoint URL] [--model ID] [--out DIR] <subcommand> [args...]",
        description=__doc__.split("\n\n", 2)[2], epilog="subcommands:\n" + listing(),
        formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--endpoint", metavar="URL", help="server base URL, no /v1",
                    default=os.environ.get("AIC_BENCH_ENDPOINT") or DEFAULT_ENDPOINT)
    ap.add_argument("--model", metavar="ID", help="served model",
                    default=os.environ.get("AIC_BENCH_MODEL") or None)
    ap.add_argument("--out", metavar="DIR", help="results root",
                    default=os.environ.get("AIC_BENCH_OUT") or DEFAULT_OUT)
    opts = ap.parse_args(own)
    if not rest:
        ap.print_help(sys.stderr)
        return 2
    bench = BY_NAME.get(rest[0])
    if bench is None:
        ap.error(f"unknown subcommand {rest[0]!r}; `aic-bench list` shows them")

    if bench.name == "list":
        print(listing())
        return 0
    root = bench_root()
    run = Run(bench.name, checked_endpoint(opts.endpoint), opts.out.rstrip("/"), opts.model,
              frozenset(f[2:] for f in GLOBAL_FLAGS if given(own, f, abbrev=False)))
    if bench.name == "env":
        try:
            model = run.model_id or first_model(run.endpoint)
        except Exception as e:  # noqa: BLE001
            model = f"(not detected: {run.endpoint}/v1/models: {e})"
        print(f"endpoint  {run.endpoint}")
        print(f"model     {model}")
        print(f"out       {run.out}")
        print(f"root      {root}")
        return 0

    cmd = build(bench, run, rest[1:], root)
    for path in run.mkdirs:
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            sys.exit(f"aic-bench: cannot create {path}: {e}; pass --out")
    execute(cmd, {**os.environ, **run.env})
    return 0


if __name__ == "__main__":
    sys.exit(main())
