#!/usr/bin/env python3
"""Let AITER's CK prebuild use more than five concurrent module builds.

With PREBUILD_KERNELS != 0, setup.py compiles the CK modules through

    prebuid_thread_num = 5
    max_jobs = os.environ.get("MAX_JOBS")
    if max_jobs is not None and max_jobs.isdigit() and int(max_jobs) > 0:
        prebuid_thread_num = min(prebuid_thread_num, int(max_jobs))
    ...
    with ThreadPoolExecutor(max_workers=prebuid_thread_num) as executor:

The literal 5 is a ceiling, not a default: `min()` means MAX_JOBS can only
lower it.  On a 128-core build allocation that leaves the step running five
modules wide -- measured at 61 minutes for ~107 modules on job 68147805, with
the node's load average sitting at 113 of 384 cores.

Make the 5 a default that AITER_PREBUILD_THREADS can raise.  An explicit value
is taken verbatim, deliberately skipping the MAX_JOBS clamp: MAX_JOBS is ninja
width *within* a module, not how many run at once, so leaving the clamp in
would pin the thread count down to the per-module width.

Do NOT compensate by dividing MAX_JOBS by the thread count.  Modules consume
the parallel work they have, not MAX_JOBS each: measured on job 68147805, five
concurrent modules produced a load average of 113 and the three-module tail
produced 98, both against MAX_JOBS=128.  That tail is three FlashAttention
modules (libmha_fwd, module_mha_fwd, module_mha_varlen_fwd) that ran for over
85 minutes as the sole critical path -- starving them of ninja width to protect
against an oversubscription that does not occur would make the slowest part of
the build dramatically slower.

Edits the source tree, so it must run after the clone and before setup.py.
"""
import os
import sys

repo = sys.argv[1] if len(sys.argv) > 1 else "/app/aiter"
target = os.path.join(repo, "setup.py")

if not os.path.isfile(target):
    print(f"[aiter-patch] ERROR: {target} not found", file=sys.stderr)
    sys.exit(1)

src = open(target).read()

old = """        prebuid_thread_num = 5
        max_jobs = os.environ.get("MAX_JOBS")
        if max_jobs is not None and max_jobs.isdigit() and int(max_jobs) > 0:
            prebuid_thread_num = min(prebuid_thread_num, int(max_jobs))
        else:
            prebuid_thread_num = min(prebuid_thread_num, getMaxJobs())
"""

# When AITER_PREBUILD_THREADS is set it is taken verbatim -- the MAX_JOBS clamp
# is skipped on that path on purpose.  MAX_JOBS is per-module ninja width, and
# the caller is expected to set it to roughly cores/threads; leaving the clamp
# in would then silently pin the thread count back down to that small number.
new = """        _aic_threads = os.environ.get("AITER_PREBUILD_THREADS", "").strip()
        if _aic_threads.isdigit() and int(_aic_threads) > 0:
            # rocm-aic: explicit value wins, un-clamped by MAX_JOBS (which
            # bounds ninja *within* a module, not how many run at once).
            prebuid_thread_num = int(_aic_threads)
        else:
            prebuid_thread_num = 5
            max_jobs = os.environ.get("MAX_JOBS")
            if max_jobs is not None and max_jobs.isdigit() and int(max_jobs) > 0:
                prebuid_thread_num = min(prebuid_thread_num, int(max_jobs))
            else:
                prebuid_thread_num = min(prebuid_thread_num, getMaxJobs())
        print(
            f"[aiter] CK prebuild: {prebuid_thread_num} concurrent module builds",
            flush=True,
        )
"""

if old not in src:
    print(
        "[aiter-patch] WARNING: prebuid_thread_num block not found — "
        "check this AITER version; leaving setup.py unmodified"
    )
    sys.exit(0)

src = src.replace(old, new, 1)
open(target, "w").write(src)
print(
    "[aiter-patch] CK prebuild thread count now honours AITER_PREBUILD_THREADS "
    f"(was pinned at 5) in {target}"
)
