#!/usr/bin/env python3
"""Add an opt-in escape hatch to AITER's FlyDSL AOT failure assertion.

PREBUILD_KERNELS != 0 makes setup.py run `aiter.aot.flydsl.common.run_aot()`,
which compiles every kernel named in AITER's tuning CSVs and ends in

    if errors:
        raise AssertionError(f"[aiter] FlyDSL AOT failures: {'; '.join(errors)}")

which fails the whole image build.

Upstream's assertion is deliberately LEFT INTACT by default: the job list is
arch-filtered (see below), so a failure is a real compile error for a kernel
this image wants, and that should fail the build.  This script only makes
AITER_FLYDSL_AOT_TOLERATE_FAILURES=1 downgrade it to a warning, for when a
build has to get through.  Only the aggregated "FlyDSL AOT failures" assertion
is covered; the hollow-kernel ("requires target feature") and strict-errata
assertions in the same file stay fatal.

History: up to v0.1.22.post1 the job list was not arch-filtered, so a gfx942
build also attempted gfx950-only kernels (FP4/MXFP4 MoE and the matching GEMM
variants) and failed on them -- on a gfx942 bake (job 68147805) 3223 of 7104
kernels "produced no kernel".  This script (then 02-flydsl-aot-filter-by-arch)
dropped the foreign-arch jobs itself.  AITER v0.1.29.post1 does that upstream:
`aot_target_archs()` / `resolve_target_arch()` in common.py make replay.run,
moe.run, mxfp4_moe.run and mega_moe.run skip every job whose arch GPU_ARCHS
does not name.  GPU_ARCHS is what the vllm Dockerfile sets to ROCM_ARCH for
single-arch builds; unset or "native" means every arch.  The filter is
therefore no longer injected here.

Edits the source tree, so it must run after the clone and before setup.py.
Exits non-zero, leaving the tree untouched, when the assertion is not found
exactly once: a silent no-op would turn the knob the Dockerfile exports into
a lie.
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

assert_pattern = re.compile(
    r"^( *)raise AssertionError\("
    r"(f\"\[aiter\] FlyDSL AOT failures: \{'; '\.join\(errors\)\}\")"
    r"\)\n",
    re.MULTILINE,
)

matches = assert_pattern.findall(src)
if len(matches) != 1 or not re.search(r"^import os$", src, re.MULTILINE):
    print(
        f"[aiter-patch] ERROR: expected exactly one FlyDSL AOT failures "
        f"assertion (found {len(matches)}) and a top-level `import os` in "
        f"{target} — this AITER version needs the patch re-ported; "
        "common.py left unmodified",
        file=sys.stderr,
    )
    sys.exit(1)

indent, expr = matches[0]
replacement = (
    f"{indent}_aic_msg = {expr}\n"
    f'{indent}if os.environ.get("AITER_FLYDSL_AOT_TOLERATE_FAILURES") == "1":\n'
    f'{indent}    print("[aiter] WARNING (tolerated): " + _aic_msg, flush=True)\n'
    f"{indent}else:\n"
    f"{indent}    raise AssertionError(_aic_msg)\n"
)
src = assert_pattern.sub(lambda _m: replacement, src, count=1)

open(target, "w").write(src)
print(
    "[aiter-patch] FlyDSL AOT failures honour "
    f"AITER_FLYDSL_AOT_TOLERATE_FAILURES in {target}"
)
