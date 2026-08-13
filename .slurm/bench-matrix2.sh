#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Second bench-serve.sh matrix: the L2 backends, TP=8, a long-context arm, and a
# re-run of the tp1 baseline.
#
# Why a second file rather than more arms in bench-matrix.sh: editing a script
# that a running bash is still reading invalidates its file handle mid-run (bash
# reads by byte offset, and on NFS the edit is a new inode -- "error reading
# input file: Stale file handle").  That is exactly how the first tp1 arm died.
# Adding arms in a new file leaves the running matrix untouched.
#
# See bench-matrix.sh for the env contract; it is identical.
set -uo pipefail
: "${AIC_DAY_DIR:?}" "${AIC_IMAGE:?}"
cd "${AIC_DAY_DIR}" || exit 1

OUT="${OUT:-${AIC_DAY_DIR}/logs/bench/results2-${SLURM_JOB_ID:-$$}.txt}"
mkdir -p "$(dirname "${OUT}")"
export HF_HOME="${HF_HOME:-/projects/hf_cache}"
BASE_ISL=2048 BASE_OSL=512 BASE_CONC="1 8 32 64 128"

run() {  # run <name> <KEY=VAL>...
    local name="$1"; shift
    if [ -n "${ARMS:-}" ] && [[ " ${ARMS} " != *" ${name} "* ]]; then return 0; fi
    printf '\n===== %s : %s =====\n' "$(date -Is)" "${name}"
    local line
    line=$(env RUN_NAME="${name}" AIC_DAY_DIR="${AIC_DAY_DIR}" AIC_IMAGE="${AIC_IMAGE}" \
        HF_HOME="${HF_HOME}" ISL="${BASE_ISL}" OSL="${BASE_OSL}" \
        CONC_LIST="${BASE_CONC}" READY_TRIES=180 "$@" \
        bash "${AIC_DAY_DIR}/.slurm/bench-serve.sh" 2>&1 | tee /dev/stderr |
        grep '^RESULT:')
    printf '%s\n' "${line:-RESULT: name=${name} status=crashed}" >> "${OUT}"
    sleep 10
}

# tp1 again: the original arm produced all five data points but lost its RESULT
# line to the stale-file-handle crash above.  This re-run is the citable one.
run tp1r TP=1 DEVS=4 CONNECTOR=0

# --- L2 backends.  AIC_L2_BACKEND is one of nixl (AIS_MT) | nixl_posix |
# local_disk | none.  "nvme" is NOT a valid value -- the first matrix passed it
# and the arm failed.  L1 is deliberately 1 GiB so KV spills into L2 rather than
# sitting in DRAM, which is the only way the L2 path is exercised at all.
run l2-nixl      TP=1 DEVS=4 CONNECTOR=1 L2=nixl       L1_GB=1 CONC_LIST="8 32 64"
run l2-posix     TP=1 DEVS=4 CONNECTOR=1 L2=nixl_posix L1_GB=1 CONC_LIST="8 32 64"
run l2-localdisk TP=1 DEVS=4 CONNECTOR=1 L2=local_disk L1_GB=1 CONC_LIST="8 32 64"

# --- TP=8.  Uses every GPU on the node, including 0-3.
run tp8 TP=8 DEVS=0,1,2,3,4,5,6,7 CONNECTOR=0

# --- long context: 8k in / 1k out, where prefill dominates and TP should pay
#     off more than it does at 2k/512.
run tp1-long ISL=8192 OSL=1024 TP=1 DEVS=4       CONNECTOR=0 CONC_LIST="1 8 32"
run tp4-long ISL=8192 OSL=1024 TP=4 DEVS=4,5,6,7 CONNECTOR=0 CONC_LIST="1 8 32"

printf '\n===== matrix2 complete: %s =====\n' "$(date -Is)"
cat "${OUT}"
