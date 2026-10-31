#!/usr/bin/env python3
"""Restrict AITER's FlyDSL AOT pass to the arch this image is built for.

PREBUILD_KERNELS != 0 makes setup.py run `aiter.aot.flydsl.common.run_aot()`,
which compiles every kernel named in AITER's tuning CSVs.  That job list is not
arch-filtered: each job's arch is derived from its row's `cu_num` via
`cu_num_to_arch(cu_num, default="gfx950")`, so a gfx942 build also attempts
gfx950-only kernels -- FP4/MXFP4 MoE and the matching GEMM variants.  They
cannot compile for gfx942, and run_aot ends in an unconditional AssertionError
that fails the whole image build.

Measured on a gfx942 bake (job 68147805): 7104 kernels attempted, 3223 failed,
every one of them "produced no kernel" -- a clean compile refusal, not an OOM
or a timeout.  CHUNK_GDN_H, which has no FP4 variants, was 288 ok / 0 failed.
The pass burned ~9 minutes of a 128-core allocation to throw 45% of its output
away.

So drop the foreign-arch jobs before compiling rather than compiling and
discarding them.  The arch comes from GPU_ARCHS, which the vllm Dockerfile sets
to ROCM_ARCH for single-arch builds; when GPU_ARCHS is unset, empty, "native"
or a multi-arch list, nothing is filtered and upstream behaviour is unchanged.

Upstream's assertion is deliberately LEFT INTACT.  Once the job list matches the
target arch, a failure is a real compile error for a kernel this image wants,
and that should fail the build.  Set AITER_FLYDSL_AOT_TOLERATE_FAILURES=1 to
downgrade it to a warning if you need to get a build through.

Edits the source tree, so it must run after the clone and before setup.py.
"""
import os
import re
import sys

repo = sys.argv[1] if len(sys.argv) > 1 else "/app/aiter"
target = os.path.join(repo, "aiter", "aot", "flydsl", "common.py")

if not os.path.isfile(target):
    print(f"[aiter-patch] ERROR: {target} not found", file=sys.stderr)
    sys.exit(1)

src = open(target).read()

# --- 1. Filter the collected job list by target arch -------------------------
anchor = """    if not all_jobs:
        print("[aiter] FlyDSL AOT: no kernels to compile, skipping")
        return
"""

if anchor not in src:
    print(
        "[aiter-patch] WARNING: run_aot job-collection block not found — "
        "check this AITER version; leaving common.py unmodified"
    )
    sys.exit(0)

injection = '''    # --- rocm-aic: keep only kernels for the arch this image targets --------
    # The tuning CSVs are arch-agnostic; jobs whose cu_num maps to another gfx
    # cannot compile here and would fail the build.  GPU_ARCHS is the target.
    _aic_arch = os.environ.get("GPU_ARCHS", "").strip()
    if _aic_arch and _aic_arch != "native" and ";" not in _aic_arch:
        _aic_before = len(all_jobs)
        all_jobs = [
            (kind, job)
            for kind, job in all_jobs
            if cu_num_to_arch(int(job.get("cu_num", 0) or 0)) == _aic_arch
        ]
        print(
            f"[aiter] FlyDSL AOT: {len(all_jobs)}/{_aic_before} kernels target "
            f"{_aic_arch}; dropped {_aic_before - len(all_jobs)} for other archs",
            flush=True,
        )

'''

src = src.replace(anchor, injection + anchor, 1)

# --- 2. Optional escape hatch on the assertion -------------------------------
assert_pattern = re.compile(
    r"( *)raise AssertionError\(\n"
    r"( *)(f\"\[aiter\] FlyDSL AOT failures \(\{tally\}\): \" \+ \"; \"\.join\(head\) \+ suffix)\n"
    r" *\)\n"
)

m = assert_pattern.search(src)
if m:
    indent, inner_indent, expr = m.group(1), m.group(2), m.group(3)
    replacement = (
        f'{indent}_aic_msg = (\n'
        f'{inner_indent}{expr}\n'
        f'{indent})\n'
        f'{indent}if os.environ.get("AITER_FLYDSL_AOT_TOLERATE_FAILURES") == "1":\n'
        f'{indent}    print("[aiter] WARNING (tolerated): " + _aic_msg, flush=True)\n'
        f'{indent}else:\n'
        f'{indent}    raise AssertionError(_aic_msg)\n'
    )
    src = src[: m.start()] + replacement + src[m.end() :]
else:
    print("[aiter-patch] WARNING: FlyDSL AOT assertion not found; not adding the escape hatch")

open(target, "w").write(src)
print(f"[aiter-patch] FlyDSL AOT job list now filtered by GPU_ARCHS in {target}")
