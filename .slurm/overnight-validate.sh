#!/usr/bin/env bash
# Unattended overnight validation of the dual-arch (gfx942;gfx950) image.
#
# Waits for a build job to finish, then runs the test suite against whatever it
# produced, one target at a time.  Sequential on purpose: every test uses the
# same fixed container names (aic-vllm-gpu0, aic-lmcache) and vllm still
# publishes a host port, so two of them on one node would collide.
#
# Survives the launching shell (run it under nohup/setsid).  Every result is
# appended to the report file as it completes, so a partial run is still
# readable if the machine goes away.
#
#   .slurm/overnight-validate.sh <build-job-id|none> <report-file>
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_JOB="${1:-none}"
REPORT="${2:-${REPO}/logs/overnight-$(date +%Y%m%d-%H%M).md}"

# The arch pair under test.  Exported so every `make` below agrees, and so the
# image tag resolves to ...-gfx942-950 rather than silently falling back to
# :latest (that quoting bug is fixed in the Makefile, but be explicit).
export ROCM_ARCH='gfx942;gfx950'
export AIC_ROCM_ARCH='gfx942;gfx950'

mkdir -p "$(dirname "${REPORT}")"
say() { printf '%s\n' "$*" | tee -a "${REPORT}"; }

say "# Overnight validation — $(date '+%Y-%m-%d %H:%M')"
say ""
say "Arch under test: \`${AIC_ROCM_ARCH}\`"
say ""

# --- 1. wait for the build ----------------------------------------------------
if [[ "${BUILD_JOB}" != "none" ]]; then
    say "## Build ${BUILD_JOB}"
    while squeue -h -j "${BUILD_JOB}" -o "%T" 2>/dev/null \
            | grep -qE "RUNNING|PENDING|CONFIGURING|COMPLETING"; do
        sleep 120
    done
    _bstate="$(sacct -j "${BUILD_JOB}" --format=State -n 2>/dev/null | head -1 | tr -d ' ')"
    _belapsed="$(sacct -j "${BUILD_JOB}" --format=Elapsed -n 2>/dev/null | head -1 | tr -d ' ')"
    say "- state: \`${_bstate}\`  elapsed: \`${_belapsed}\`"
    # The exit file is the truth: a requeued job can report PENDING for work
    # that already succeeded (tiny-test 68160869 reached Restarts=2 at exit 0).
    _bexit="$(cat "${REPO}/logs/${BUILD_JOB}/build.exit" 2>/dev/null || echo '?')"
    say "- build.exit: \`${_bexit}\`"
    if [[ "${_bexit}" != "0" ]]; then
        say ""
        say "Build did not succeed; running the tests anyway against whatever image"
        say "is already on /scratch, and saying so rather than skipping silently."
    fi
    say ""
fi

# --- 2. run each target, record, continue on failure --------------------------
run_target() {
    local target="$1" logname="$2"
    say "## ${target}"
    local t0 t1 out jobid
    t0="$(date +%s)"
    out="$(cd "${REPO}" && make "${target}" 2>&1)" || true
    t1="$(date +%s)"
    jobid="$(printf '%s\n' "${out}" | grep -oE 'submitted [a-z-]+ as job [0-9]+' | grep -oE '[0-9]+$' | tail -1)"
    say "- wall clock: $(( (t1 - t0) / 60 ))m"
    if [[ -z "${jobid}" ]]; then
        say "- **no job id parsed** — submission itself failed:"
        say '```'
        printf '%s\n' "${out}" | tail -15 | tee -a "${REPORT}" >/dev/null
        say '```'
        say ""
        return 1
    fi
    say "- job: \`${jobid}\`"
    local state exitf
    state="$(sacct -j "${jobid}" --format=State -n 2>/dev/null | head -1 | tr -d ' ')"
    exitf="$(cat "${REPO}/logs/${jobid}/${logname}.exit" 2>/dev/null || echo '?')"
    say "- state: \`${state}\`  ${logname}.exit: \`${exitf}\`"
    local L="${REPO}/logs/${jobid}/${logname}.out"
    if [[ -s "${L}" ]]; then
        say "- key lines:"
        say '```'
        sed 's/\x1b\[[0-9;]*m//g' "${L}" \
            | grep -iE "ALL CHECKS PASSED|ALL PHASES PASSED|non-empty completion|OK Phase|FAIL|ERROR:|Traceback|up=1 job=|TSDB:" \
            | tail -25 | tee -a "${REPORT}" >/dev/null
        say '```'
        local tsdb
        tsdb="$(du -sh "${REPO}/logs/${jobid}/prometheus" 2>/dev/null | cut -f1)"
        say "- TSDB: \`${tsdb:-absent}\`"
    else
        say "- job log empty or missing (job may never have started)"
    fi
    say ""
    [[ "${exitf}" == "0" ]]
}

PASS=0; FAIL=0
for spec in "smoke-test:smoke-test" "tiny-test:tiny-test" "accuracy-test:accuracy-test"; do
    if run_target "${spec%%:*}" "${spec##*:}"; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
done

say "## Summary"
say ""
say "- passed: ${PASS}"
say "- failed: ${FAIL}"
say "- finished: $(date '+%Y-%m-%d %H:%M')"
