#!/usr/bin/env bash
# Watch a run-rocjitsu-cpu.sbatch job: emit on step change, on any failure
# signal, and once at completion with a summary.
#
# The run takes ~26 min in probe mode and hours with the vLLM arm, and [6d]
# buffers its output, so without this the job looks identical whether it is
# making progress or wedged.
#
#   .slurm/watch-rocjitsu.sh <job-id>
#
# RJ_LOG_DIR overrides where the job log lives (matches the --output= passed
# to sbatch; the default is the logs/rocjitsu-%j.log this repo submits with).
set -uo pipefail

JOB="${1:?usage: watch-rocjitsu.sh <job-id>}"
LOG="${RJ_LOG_DIR:-logs}/rocjitsu-${JOB}.log"
POLL_S="${RJ_WATCH_POLL_S:-120}"

# Lines worth interrupting for.  vllm_reset_test_rc= covers the vLLM arm;
# "could not pull" covers the registry path falling back to a local build.
SIG_RE='^FAIL|Permission denied|^vllm_reset_test_rc=|could not pull|CEILING'
# Step markers: [N/6] for the outer phases, [6a]..[6d] and [6c.1] for the
# inner ones.
STEP_RE='^  \[6[abcd](\.[0-9])?\][^.]*|^\[[0-9]/6\][^.]*'

last_step=""
last_sig=0
while true; do
    # Alive = present in squeue in ANY state.  Matching only RUNNING made a
    # job that was briefly PENDING at submission look like it had finished.
    if [ -z "$(squeue -j "${JOB}" -h -o '%T' 2>/dev/null)" ]; then
        echo "JOB ENDED: $(sacct -j "${JOB}" --format=State,ExitCode -n 2>/dev/null | head -1 | tr -s ' ')"
        grep -aE "^\[[0-9]/6\]|^  \[6[abcd]|${SIG_RE}|GPU check done|available=True|rungs passed|=== (PASS|FAIL)|exited [0-9]+ ===" \
            "${LOG}" 2>/dev/null | tail -25
        break
    fi

    step=$(grep -aoE "${STEP_RE}" "${LOG}" 2>/dev/null | tail -1)
    if [ "${step}" != "${last_step}" ]; then
        echo "step $(date +%H:%M) | ${step}"
        last_step="${step}"
    fi

    n=$(grep -acE "${SIG_RE}" "${LOG}" 2>/dev/null || echo 0)
    if [ "${n:-0}" -gt "${last_sig}" ]; then
        grep -aE "${SIG_RE}" "${LOG}" | tail -4 | sed 's/^/SIGNAL: /'
        last_sig="${n}"
    fi

    sleep "${POLL_S}"
done
