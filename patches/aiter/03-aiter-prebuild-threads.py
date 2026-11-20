#!/usr/bin/env python3
"""Let AITER's CK prebuild use more than five concurrent module builds.

With PREBUILD_KERNELS != 0, setup.py compiles the CK modules through

    outer_workers, inner_workers = split_worker_budget(
        get_worker_count(), len(all_opts_args_build)
    )
    ...
    with (
        worker_ceiling(inner_workers),
        ThreadPoolExecutor(max_workers=outer_workers) as executor,
    ):

(v0.1.29.post1; up to v0.1.22.post1 the same thing was spelled
`prebuid_thread_num = 5` clamped by MAX_JOBS.)  split_worker_budget() caps the
outer pool at aiter_worker_limits.MAX_OUTER_MODULE_WORKERS = 5 and gives each
module total // outer Ninja jobs.  The 5 is a ceiling, not a default: the
worker budget can only lower it.  On a 128-core build allocation that leaves
the step running five modules wide -- measured at 61 minutes for ~107 modules
on job 68147805 (v0.1.22.post1), with the node's load average sitting at 113 of
384 cores.

Make the 5 a default that AITER_PREBUILD_THREADS can raise.  An explicit value
is taken verbatim (bounded only by the number of modules), deliberately
skipping the worker-budget clamp: that budget (MAX_JOBS, which setup.py adopts
as AITER_MAX_JOBS, further limited by AITER to 80% of the CPUs and to available
memory) is ninja width *within* a module, not how many run at once, so leaving
the clamp in would pin the thread count down to the per-module width.

Do NOT compensate by dividing the budget by the thread count, which is what
upstream's split now does on its own path.  Modules consume the parallel work
they have, not MAX_JOBS each: measured on job 68147805, five concurrent modules
produced a load average of 113 and the three-module tail produced 98, both
against MAX_JOBS=128.  That tail is three FlashAttention modules (libmha_fwd,
module_mha_fwd, module_mha_varlen_fwd) that ran for over 85 minutes as the sole
critical path -- starving them of ninja width to protect against an
oversubscription that does not occur would make the slowest part of the build
dramatically slower.  So with AITER_PREBUILD_THREADS set, every module keeps
the whole worker budget; without it, upstream's split is untouched.

Edits the source tree, so it must run after the clone and before setup.py.
Exits non-zero, leaving the tree untouched, when the block is not found.
"""
import os
import sys

repo = sys.argv[1] if len(sys.argv) > 1 else "/app/aiter"
target = os.path.join(repo, "setup.py")

if not os.path.isfile(target):
    print(f"[aiter-patch] ERROR: {target} not found", file=sys.stderr)
    sys.exit(1)

src = open(target).read()

old = """        outer_workers, inner_workers = split_worker_budget(
            get_worker_count(), len(all_opts_args_build)
        )
"""

# When AITER_PREBUILD_THREADS is set it is taken verbatim -- the budget split
# is skipped on that path on purpose.  The worker budget is per-module ninja
# width; dividing it by the thread count would starve the long MHA tail.
new = """        _aic_threads = os.environ.get("AITER_PREBUILD_THREADS", "").strip()
        if _aic_threads.isdigit() and int(_aic_threads) > 0:
            # rocm-aic: explicit value wins, not capped at
            # MAX_OUTER_MODULE_WORKERS, and each module keeps the whole worker
            # budget (it bounds ninja *within* a module, not how many run at
            # once).
            outer_workers = max(1, min(int(_aic_threads), len(all_opts_args_build)))
            inner_workers = get_worker_count()
        else:
            outer_workers, inner_workers = split_worker_budget(
                get_worker_count(), len(all_opts_args_build)
            )
        print(
            f"[aiter] CK prebuild: {outer_workers} concurrent module builds, "
            f"ninja -j{inner_workers} each",
            flush=True,
        )
"""

if src.count(old) != 1 or "ThreadPoolExecutor(max_workers=outer_workers)" not in src:
    print(
        "[aiter-patch] ERROR: split_worker_budget block not found exactly "
        f"once in {target} — this AITER version needs the patch re-ported; "
        "setup.py left unmodified",
        file=sys.stderr,
    )
    sys.exit(1)

src = src.replace(old, new, 1)
open(target, "w").write(src)
print(
    "[aiter-patch] CK prebuild thread count now honours AITER_PREBUILD_THREADS "
    f"(was capped at 5) in {target}"
)
