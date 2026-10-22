#!/usr/bin/env python3
"""Stop AITER's FlyDSL AOT pass from failing a single-arch build.

PREBUILD_KERNELS != 0 makes setup.py run `aiter.aot.flydsl.common.run_aot()`
before the CK kernel builds.  run_aot compiles every kernel named in AITER's
tuning CSVs, and those CSVs are not arch-filtered: the arch per job comes from
the row's `cu_num` via `cu_num_to_arch(..., default="gfx950")`, so a gfx942
build still tries to compile gfx950-only kernels -- FP4/MXFP4 MoE and the
matching GEMM variants.  Those produce no kernel, and run_aot ends with an
unconditional `raise AssertionError`, which fails the whole image build.

Measured on a gfx942 bake (job 68147390): 7104 kernels attempted, 3223 failed,
all of them "produced no kernel" (a clean compile refusal, not an OOM or a
timeout -- the distinction matters, because it rules out worker sizing as the
cause).  CHUNK_GDN_H, which has no FP4 variants, was 288 ok / 0 failed.

There is no upstream knob for this: get_max_workers/get_kernel_timeout/
get_max_retries are all tunable by env var, but the assertion is not.  So turn
it into a warning.  Kernels that did compile are kept and the CK prebuild --
which is the part that removes the vLLM cold-start JIT -- proceeds.

Run AFTER `pip install` of the aiter wheel is NOT possible here: the AOT pass
happens during `setup.py bdist_wheel`.  This patch therefore edits the source
tree, and must run after the clone and before setup.py.
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

# The assertion is built over several lines and ends with the joined error
# list; match from `raise AssertionError(` to the closing paren of that call.
pattern = re.compile(
    r"( *)raise AssertionError\(\n"
    r' *f"\[aiter\] FlyDSL AOT failures \(\{tally\}\): " \+ "; "\.join\(head\) \+ suffix\n'
    r" *\)\n"
)

match = pattern.search(src)
if not match:
    print(
        "[aiter-patch] WARNING: FlyDSL AOT assertion not found — "
        "check this AITER version; leaving common.py unmodified"
    )
    sys.exit(0)

indent = match.group(1)
replacement = (
    f'{indent}print(\n'
    f'{indent}    f"[aiter] WARNING: FlyDSL AOT partial failure ({{tally}}): "\n'
    f'{indent}    + "; ".join(head)\n'
    f'{indent}    + suffix,\n'
    f'{indent}    flush=True,\n'
    f'{indent})\n'
    f'{indent}print(\n'
    f'{indent}    "[aiter] continuing: AITER ships kernels for archs this build "\n'
    f'{indent}    "does not target, and those cannot compile here. Kernels that "\n'
    f'{indent}    "did compile are kept; anything missing falls back to JIT.",\n'
    f'{indent}    flush=True,\n'
    f'{indent})\n'
)

src = src[: match.start()] + replacement + src[match.end() :]
open(target, "w").write(src)
print(f"[aiter-patch] FlyDSL AOT failures downgraded to a warning in {target}")
