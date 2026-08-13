#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Correctness gate: does the rebased tree still retrieve KV from LMCache L1 and
# L2?  This is `make vllm-reset-test` -- the repo's own end-to-end retrieval
# test (benchmarks/vllm_reset_test.py) -- wrapped with the node-specific paths
# and the HF token, and run once per L2 backend.
#
# Throughput arms deliberately do NOT answer this question: they send unique
# random prompts, so nothing is ever retrieved and a completely dead L2 would
# score identically.  Only this test resets the GPU prefix cache and then checks
# LMCache's own hit counters.
#
# Env:
#   AIC_DAY_DIR    repo root                       (required)
#   HF_TOKEN       HF read token, or HF_TOKEN_FILE pointing at a file holding it
#   L2_BACKENDS    backends to gate (default: nixl_posix local_disk nixl)
#   AIC_TEST_FLOOD number of flood prompts (default: 150)
#   GATE_L1_GB     LMCache L1 size (default: 0.25 -- see below)
#
# Why L1 is 0.25 GiB and not 1 GiB.  The test's premise is "flood L1 until the
# anchors are evicted, then prove they come back from L2".  Measured on this node,
# that premise does not hold at 1 GiB: with 150 and with 400 flood prompts -- 1234
# and 3300+ chunks of unique KV against a 222-chunk cap -- L1 occupancy plateaus at
# 70-75 % (698-752 MiB) and never reaches the cap, so nothing is ever evicted and
# all 63 anchor chunks are still served from L1.  LMCache is admitting only a
# fraction of the flood into L1 while storing all of it to L2.  Flooding harder
# cannot fix this; the plateau is below the cap by construction.
#
# So size L1 under the anchor set instead: 10 anchors = 63 chunks at ~4.5 MiB =
# ~284 MiB, which does not fit in 0.25 GiB.  Now the anchors themselves force
# eviction and the L2 read path is genuinely exercised.
set -uo pipefail
: "${AIC_DAY_DIR:?}"
cd "${AIC_DAY_DIR}" || exit 1

L2_BACKENDS="${L2_BACKENDS:-nixl_posix local_disk nixl}"
export HF_HOME="${HF_HOME:-/projects/hf_cache}"

# Same contract as the Makefile's _check_hf_token and run-build-distribute.sh:
# HF_TOKEN, or HF_TOKEN_FILE pointing at it.  No default path -- the token file
# is per-user and naming one here would only work for whoever wrote it.
if [ -z "${HF_TOKEN:-}" ] && [ -n "${HF_TOKEN_FILE:-}" ] && [ -r "${HF_TOKEN_FILE}" ]; then
    HF_TOKEN="$(tr -d ' \t\n\r' < "${HF_TOKEN_FILE}")"
    export HF_TOKEN
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "l2-gate: set HF_TOKEN or HF_TOKEN_FILE; make vllm-reset-test gates on it" >&2
    exit 1
fi

NV=""
for d in /mnt/nixl-nvme-0 /mnt/nixl-nvme-* /mnt/lmcache-nvme; do
    [ -d "$d" ] && [ -w "$d" ] && { NV="$d"; break; }
done
[ -n "${NV}" ] || { echo "l2-gate: no writable NVMe mount" >&2; exit 1; }

OUT="${AIC_DAY_DIR}/logs/bench/l2-gate-${SLURM_JOB_ID:-$$}.txt"
mkdir -p "$(dirname "${OUT}")"
rc_all=0

for be in ${L2_BACKENDS}; do
    pool="${NV}/aic-l2gate-${SLURM_JOB_ID:-$$}-${be}"
    mkdir -p "${pool}"
    log="${AIC_DAY_DIR}/logs/bench/l2-gate-${be}.log"
    echo "=== l2-gate ${be} $(date -Is)"

    # Each backend gets a clean pool dir, so a hit can only come from KV this
    # invocation wrote -- never from a previous backend's leftovers.
    # NFS_DATA as well as NVME_DATA: _prep_dirs mkdir -p's both, and the default
    # /mnt/lmcache-nfs is not writable on this node -- make dies at the first
    # prerequisite, long before the test runs.
    # The AIS_MT backend cannot register its buffers on this node's ext4 NVMe
    # (hipFileBufRegister err=5013) and takes the whole stack down with it.  Compat
    # mode is the fallback the error message asks for; it costs the P2PDMA
    # fastpath, so it is set only for this backend and only to get a verdict.
    _compat=false; [ "${be}" = "nixl" ] && _compat=true

    env AIC_L2_BACKEND="${be}" \
        HIPFILE_ALLOW_COMPAT_MODE="${_compat}" \
        LMCACHE_L1_SIZE_GB="${GATE_L1_GB:-0.25}" \
        AIC_TEST_FLOOD="${AIC_TEST_FLOOD:-150}" \
        NVME_DATA="${pool}" \
        NFS_DATA="${pool}-nfs" \
        LOG="${AIC_DAY_DIR}/logs/bench/l2-gate-${be}-logs" \
        VLLM_MODEL="Qwen/Qwen2.5-3B-Instruct" \
        AIC_VISIBLE_DEVICES=4 \
        AIC_RESET_MONITORING=0 \
        make -C "${AIC_DAY_DIR}" vllm-reset-test > "${log}" 2>&1
    rc=$?
    tail -30 "${log}"

    # Do NOT read the exit code as a verdict.  vllm_reset_test.py exits 4 when L2
    # retrieval fails, but it runs under make, and make normalises every recipe
    # failure to 2 -- so the code says "2" whatever went wrong, including make
    # dying in a prerequisite before the test started.  The log lines are the only
    # trustworthy signal.
    l1="$(grep -c 'L1 retrieval PASS' "${log}")"
    l2="$(grep -c 'L2 retrieval PASS' "${log}")"
    if ! grep -q 'LMCache L1 + L2 retrieval verification' "${log}"; then
        verdict="FAIL: harness error, test never ran -- $(grep -m1 -iE 'error|denied|no such' "${log}" | tail -c 120)"
        ok=0
    elif [ "${l2}" -gt 0 ]; then
        verdict="pass (L2 retrieved$([ "${l1}" -gt 0 ] && echo ', L1 too'))"
        ok=1
    elif [ "${l1}" -gt 0 ]; then
        # L1 served every anchor, so nothing was ever evicted and the L2 read path
        # was never exercised.  That is an under-sized test, not a broken backend.
        # Report whether L2 was at least WRITTEN -- "stored > 0 but no hits" is a
        # read-path suspicion, "stored == 0" points at the store path (or, for
        # local_disk, at the l2_name label the test hardcodes; see §6.10).
        # Anchor the digits to end-of-match, or the "2" in "L2" is captured too.
        _stored="$(grep -oE 'L2 stored +[0-9]+' "${log}" | tail -1 | grep -oE '[0-9]+$')"
        verdict="INCONCLUSIVE: L1 hits only, anchors never evicted (L2 stored=${_stored:-?} chunks); lower GATE_L1_GB"
        ok=0
    else
        verdict="FAIL: no hits in either tier"
        ok=0
    fi

    echo "L2GATE: backend=${be} rc=${rc} ${verdict}" | tee -a "${OUT}"
    [ "${ok}" = "1" ] || rc_all=1

    make -C "${AIC_DAY_DIR}" down >/dev/null 2>&1 || true
    for c in aic-vllm-gpu0 aic-lmcache aic-lmcache-coordinator aic-client; do
        docker rm -f "$c" >/dev/null 2>&1 || true
    done
    rm -rf "${pool}" 2>/dev/null || true
    sleep 10
done

echo "=== l2-gate summary"; cat "${OUT}"
exit "${rc_all}"
