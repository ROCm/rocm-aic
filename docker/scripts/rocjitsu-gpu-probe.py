#!/usr/bin/env python3
"""Characterise what an emulated GPU can actually execute.

[6c] of test-rocjitsu-local only enumerates the device -- it never launches a
kernel -- so a run that gets `available=True count=1 arch=gfx1250` and then
stalls forever inside vLLM leaves no way to tell "emulation is slow" from
"emulation cannot dispatch".  This walks a ladder from device properties up to
the kernels vLLM actually issues at startup, and reports where the ladder ends.

Each rung runs in its own subprocess under a wall-clock timeout.  That matters:
a kernel wedged in the driver does not respond to Python-level interrupts, so
an in-process alarm would hang the probe along with it.  A fresh process per
rung also means one wedged rung does not poison the rest.

Usage:
    rocjitsu-gpu-probe.py                  # run the ladder, print a summary
    rocjitsu-gpu-probe.py --timeout 120    # per-rung wall clock (default 300)
    rocjitsu-gpu-probe.py --only add_small,gemm_small
    rocjitsu-gpu-probe.py --run add_small  # internal: execute one rung

Exit status is 0 if every rung passed, 1 otherwise.  A `timeout` or `fail`
result is the answer, not an error -- the point is to find the ceiling.
"""

import argparse
import json
import os
import subprocess
import sys
import time

# Rungs run in this order, cheapest and most fundamental first.  Ordering is
# deliberate: each one is a prerequisite of those below it, so the first
# non-ok result is the ceiling and everything after it is noise.
LADDER = [
    "enum",
    "alloc",
    "h2d",
    "d2h",
    "fill",
    "add_small",
    "reduce",
    "gemm_small",
    "softmax",
    "layernorm",
    "add_large",
    "gemm_large",
    "sdpa",
    "triton",
    "memprofile",
]

# Rungs whose failure does not invalidate the ones below them.  triton is the
# backend vLLM selects on ROCm (TRITON_ATTN), so its result is reported but a
# JIT toolchain problem is a different fault from the device being unable to
# run a kernel at all.
SOFT = {"triton"}


def _dev():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    return torch.device("cuda:0")


# --- rungs -----------------------------------------------------------------
# Each returns a short detail string.  Anything that raises is a failure.
# Every rung that produces a value verifies it against the CPU result: a
# kernel that launches and returns garbage is a different (and worse) finding
# than one that does not launch, and silent wrong answers are exactly what a
# half-implemented emulator produces.


def rung_enum():
    import torch

    d = _dev()
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(d)
    return (
        f"arch={p.gcnArchName} name={p.name} "
        f"cus={p.multi_processor_count} "
        f"vram_total={total / 2**20:.0f}MiB vram_free={free / 2**20:.0f}MiB"
    )


def rung_alloc():
    """hipMalloc only -- no kernel, no copy."""
    import torch

    d = _dev()
    x = torch.empty(1024, dtype=torch.float32, device=d)
    torch.cuda.synchronize()
    return f"allocated {x.numel() * 4} bytes at {hex(x.data_ptr())}"


def rung_h2d():
    """Host->device DMA.  No compute kernel involved."""
    import torch

    d = _dev()
    host = torch.arange(262144, dtype=torch.float32)
    t0 = time.perf_counter()
    dev = host.to(d)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    mb = host.numel() * 4 / 2**20
    return f"{mb:.2f}MiB in {dt:.3f}s ({mb / dt:.2f} MiB/s)"


def rung_d2h():
    """Round trip, verified.  Still no compute kernel -- copies only."""
    import torch

    d = _dev()
    host = torch.arange(262144, dtype=torch.float32)
    back = host.to(d).cpu()
    if not torch.equal(host, back):
        bad = int((host != back).sum())
        raise RuntimeError(f"round trip corrupted {bad}/{host.numel()} elements")
    return f"{host.numel()} elements survived the round trip intact"


def rung_fill():
    """First real kernel dispatch: a device-side fill, then verify on host."""
    import torch

    d = _dev()
    t0 = time.perf_counter()
    x = torch.full((4096,), 3.5, dtype=torch.float32, device=d)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    got = x.cpu()
    if not torch.all(got == 3.5):
        raise RuntimeError(f"fill produced wrong values, e.g. {got[:8].tolist()}")
    return f"fill kernel ran and verified in {dt:.3f}s"


def rung_add_small():
    """Elementwise add, 1Ki elements -- the simplest arithmetic kernel."""
    import torch

    d = _dev()
    a = torch.arange(1024, dtype=torch.float32)
    b = torch.ones(1024, dtype=torch.float32)
    t0 = time.perf_counter()
    c = (a.to(d) + b.to(d)).cpu()
    dt = time.perf_counter() - t0
    if not torch.allclose(c, a + b):
        raise RuntimeError("add produced wrong values")
    return f"1Ki-element add verified in {dt:.3f}s"


def rung_reduce():
    """Cross-workgroup reduction -- needs more than one block to cooperate."""
    import torch

    d = _dev()
    a = torch.ones(1 << 20, dtype=torch.float32)
    t0 = time.perf_counter()
    got = float(a.to(d).sum().cpu())
    dt = time.perf_counter() - t0
    if abs(got - float(1 << 20)) > 1.0:
        raise RuntimeError(f"sum wrong: got {got}, want {1 << 20}")
    return f"1Mi-element sum = {got:.0f}, verified in {dt:.3f}s"


def _gemm(n, tol):
    import torch

    d = _dev()
    a = torch.randn(n, n, dtype=torch.float32)
    b = torch.randn(n, n, dtype=torch.float32)
    t0 = time.perf_counter()
    got = (a.to(d) @ b.to(d)).cpu()
    dt = time.perf_counter() - t0
    err = float((got - (a @ b)).abs().max())
    if err > tol:
        raise RuntimeError(f"{n}x{n} gemm max abs err {err:.4g} > {tol}")
    gflop = 2 * n**3 / 1e9
    return f"{n}x{n} gemm in {dt:.3f}s ({gflop / dt:.4f} GFLOP/s) max_err={err:.3g}"


def rung_gemm_small():
    """rocBLAS/hipBLASLt dispatch -- a different path from torch's own kernels."""
    return _gemm(64, 1e-2)


def rung_gemm_large():
    """Same path at a size that shows whether throughput scales at all."""
    return _gemm(1024, 1e-1)


def rung_softmax():
    import torch

    d = _dev()
    x = torch.randn(8, 4096, dtype=torch.float32)
    t0 = time.perf_counter()
    got = torch.softmax(x.to(d), dim=-1).cpu()
    dt = time.perf_counter() - t0
    err = float((got - torch.softmax(x, dim=-1)).abs().max())
    if err > 1e-4:
        raise RuntimeError(f"softmax max abs err {err:.4g}")
    return f"softmax(8,4096) in {dt:.3f}s max_err={err:.3g}"


def rung_layernorm():
    import torch

    d = _dev()
    x = torch.randn(8, 4096, dtype=torch.float32)
    t0 = time.perf_counter()
    got = torch.nn.functional.layer_norm(x.to(d), (4096,)).cpu()
    dt = time.perf_counter() - t0
    err = float((got - torch.nn.functional.layer_norm(x, (4096,))).abs().max())
    if err > 1e-3:
        raise RuntimeError(f"layer_norm max abs err {err:.4g}")
    return f"layer_norm(8,4096) in {dt:.3f}s max_err={err:.3g}"


def rung_add_large():
    """4Mi elements -- gives a bandwidth number to extrapolate vLLM from."""
    import torch

    d = _dev()
    n = 1 << 22
    a = torch.ones(n, dtype=torch.float32, device=d)
    b = torch.ones(n, dtype=torch.float32, device=d)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    c = a + b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if float(c[0].cpu()) != 2.0:
        raise RuntimeError("large add produced wrong values")
    gb = 3 * n * 4 / 2**30
    return f"4Mi-element add in {dt:.3f}s ({gb / dt:.4f} GiB/s effective)"


def rung_sdpa():
    """Attention through torch's own SDPA -- the shape vLLM's prefill issues."""
    import torch

    d = _dev()
    q = torch.randn(1, 4, 128, 64, dtype=torch.float32)
    t0 = time.perf_counter()
    got = torch.nn.functional.scaled_dot_product_attention(
        q.to(d), q.to(d), q.to(d)
    ).cpu()
    dt = time.perf_counter() - t0
    ref = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    err = float((got - ref).abs().max())
    if err > 1e-3:
        raise RuntimeError(f"sdpa max abs err {err:.4g}")
    return f"sdpa(1,4,128,64) in {dt:.3f}s max_err={err:.3g}"


def rung_triton():
    """Triton JIT compile + launch.

    vLLM selects TRITON_ATTN on ROCm, so if Triton cannot compile for this
    target the engine cannot start no matter how fast the device is.  This
    exercises compile and launch separately so the timing says which half is
    expensive.
    """
    import torch
    import triton
    import triton.language as tl

    d = _dev()

    @triton.jit
    def _add1(src, dst, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        tl.store(dst + off, tl.load(src + off, mask=mask) + 1.0, mask=mask)

    n = 4096
    src = torch.arange(n, dtype=torch.float32, device=d)
    dst = torch.empty_like(src)
    t0 = time.perf_counter()
    _add1[(n // 256,)](src, dst, n, BLOCK=256)
    torch.cuda.synchronize()
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    _add1[(n // 256,)](src, dst, n, BLOCK=256)
    torch.cuda.synchronize()
    t_cached = time.perf_counter() - t0

    got = dst.cpu()
    if not torch.allclose(got, torch.arange(n, dtype=torch.float32) + 1.0):
        raise RuntimeError(f"triton kernel wrong, e.g. {got[:8].tolist()}")
    return (
        f"triton {triton.__version__} compile+launch {t_first:.3f}s, "
        f"cached launch {t_cached:.3f}s, verified"
    )


def rung_memprofile():
    """What vLLM does during startup: measure free VRAM across a big alloc.

    vLLM's memory profiling runs a forward pass and reads mem_get_info on
    either side.  An emulator that reports a static free figure makes vLLM
    compute a nonsense KV-cache size, which is its own failure mode distinct
    from a kernel hang.
    """
    import torch

    d = _dev()
    free0, total = torch.cuda.mem_get_info(d)
    block = torch.empty(64 << 20, dtype=torch.uint8, device=d)  # 64 MiB
    torch.cuda.synchronize()
    free1, _ = torch.cuda.mem_get_info(d)
    delta = (free0 - free1) / 2**20
    del block
    torch.cuda.empty_cache()
    free2, _ = torch.cuda.mem_get_info(d)
    if delta < 1:
        raise RuntimeError(
            f"free VRAM did not move across a 64MiB alloc "
            f"(before={free0 / 2**20:.0f}MiB after={free1 / 2**20:.0f}MiB) -- "
            f"vLLM would size its KV cache from this"
        )
    return (
        f"total={total / 2**20:.0f}MiB free {free0 / 2**20:.0f}->"
        f"{free1 / 2**20:.0f}->{free2 / 2**20:.0f}MiB, 64MiB alloc moved it {delta:.0f}MiB"
    )


RUNGS = {name: globals()[f"rung_{name}"] for name in LADDER}


# --- driver ----------------------------------------------------------------


def run_one(name):
    """Execute a single rung in this process and emit a JSON result line."""
    t0 = time.perf_counter()
    try:
        detail = RUNGS[name]()
        out = {"status": "ok", "detail": detail}
    except Exception as exc:  # noqa: BLE001 - any failure is a result, not a crash
        out = {"status": "fail", "detail": f"{type(exc).__name__}: {exc}"}
    out["name"] = name
    out["elapsed"] = round(time.perf_counter() - t0, 3)
    print("PROBE_JSON " + json.dumps(out), flush=True)
    return 0 if out["status"] == "ok" else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="execute one rung in this process (internal)")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="per-rung wall clock, seconds (default 300)")
    ap.add_argument("--only", help="comma-separated subset of rungs to run")
    ap.add_argument("--keep-going", action="store_true",
                    help="run every rung even after one fails")
    args = ap.parse_args()

    if args.run:
        return run_one(args.run)

    names = LADDER
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
        unknown = [n for n in names if n not in RUNGS]
        if unknown:
            print(f"unknown rung(s): {', '.join(unknown)}", file=sys.stderr)
            return 2

    print(f"=== rocjitsu GPU probe: {len(names)} rungs, {args.timeout:g}s each ===",
          flush=True)
    results = []
    ceiling = None
    for name in names:
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", os.path.abspath(__file__), "--run", name],
                capture_output=True, text=True, timeout=args.timeout,
            )
            res = None
            for line in proc.stdout.splitlines():
                if line.startswith("PROBE_JSON "):
                    res = json.loads(line[len("PROBE_JSON "):])
            if res is None:
                # The child died before reporting -- a segfault or an abort in
                # the driver looks like this, and is itself a finding.
                tail = (proc.stderr.strip().splitlines() or ["(no stderr)"])[-1]
                res = {"name": name, "status": "crash",
                       "elapsed": round(time.perf_counter() - t0, 3),
                       "detail": f"exit={proc.returncode}: {tail}"}
        except subprocess.TimeoutExpired:
            res = {"name": name, "status": "timeout",
                   "elapsed": round(time.perf_counter() - t0, 3),
                   "detail": f"no result within {args.timeout:g}s (killed)"}

        results.append(res)
        print(f"  [{res['status']:>7}] {res['name']:<12} "
              f"{res['elapsed']:>8.3f}s  {res['detail']}", flush=True)

        if res["status"] != "ok" and name not in SOFT:
            ceiling = ceiling or name
            if not args.keep_going:
                print(f"  (stopping: '{name}' is a prerequisite of the rungs below "
                      f"it; --keep-going to run them anyway)", flush=True)
                break

    ok = [r for r in results if r["status"] == "ok"]
    print(f"=== {len(ok)}/{len(results)} rungs passed ===", flush=True)
    if ceiling:
        print(f"=== CEILING: {ceiling} ===", flush=True)
    else:
        soft_bad = [r["name"] for r in results if r["status"] != "ok"]
        if soft_bad:
            print(f"=== all hard rungs passed; soft failures: "
                  f"{', '.join(soft_bad)} ===", flush=True)
        elif set(names) == set(LADDER):
            print("=== CEILING: none -- the device ran every kernel in the ladder ===",
                  flush=True)
    print("PROBE_SUMMARY " + json.dumps({"results": results, "ceiling": ceiling}),
          flush=True)
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
