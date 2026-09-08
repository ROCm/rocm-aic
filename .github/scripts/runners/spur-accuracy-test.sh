#!/usr/bin/env bash
set -euo pipefail

# Runs on the self-hosted runner; SSHes to the SPUR head node (AIC_SPUR_HOST) and
# runs accuracy-test against the tarball produced by spur-dist-build.sh for the
# same SHA (the stage after spur-tiny-test.sh in the nightly chain, or triggered
# on demand via /run-ci-accuracy or a manual workflow_dispatch).
#
# The two accepted targets are the same gate: accuracy-test-fast is
# accuracy-test with AIC_ROCM_ARCH pinned to AIC_FAST_ARCH, so it matches the
# tarball a dist-build-fast produced.  Neither is cheaper than the other.
#
# accuracy-test is the KV-integrity gate: it scores gsm8k against a VRAM-only arm
# and a tiered (LMCache + NIXL POSIX NVMe) arm in the same job and asserts that
# routing KV through DRAM/NVMe did not change the answers.  There are no
# committed golden files -- the oracle is the same-run difference between the two
# arms, plus an absolute floor from tests/accuracy/expected.json.  See
# tests/accuracy/README.md for what each assertion catches.
#
# Cleanup ownership depends on whether another stage follows:
#   * On-demand flow: accuracy-test is terminal, so it owns the final cleanup
#     (removes the clone + tarball on exit).
#   * Nightly chain: the cliff job needs the artifacts and runs next whatever
#     this gate concludes, so the nightly step sets KEEP_ARTIFACTS=1 and this
#     script leaves them alone even on failure (spur-cliff-harvest.sh does the
#     final cleanup).  That is why the tarball is kept on failure here, unlike
#     spur-tiny-test.sh: nothing downstream of tiny-test runs on a red gate.
#
# The clone deliberately is NOT run-key scoped, unlike spur-tiny-test.sh's: the
# nightly hands it to spur-cliff{,-harvest}.sh, which still resolve
# $HOME/Projects/rocm-aic.<short-sha>.  Scoping it here would leak a clone per
# run because nothing downstream would find it to delete.
#
# Log harvesting is independent of that ownership split and happens on both
# paths: the per-job scores and per-arm container logs live under the clone,
# which one side or the other always deletes, so they are archived off it and
# scp'd back here for the workflow to upload.  A red gate is exactly when they
# matter, so neither the archive nor the fetch is conditional on success.
#
# The scores themselves are then lifted out of the archive and reported to the
# step summary (and as step outputs, which the PR chain puts in its comment).
# The gate has no committed golden, so a run's own numbers are the only record
# it produces, and an artifact nobody downloads is not a record.
# The model uses the cluster-wide HF cache so it is downloaded once and reused
# across CI workflows and SPUR accounts.

SHA="${1:?usage: $0 <full-sha> [accuracy-test|accuracy-test-fast]}"
AIC_ACCURACY_TEST_TARGET="${2:-accuracy-test}"
case "${AIC_ACCURACY_TEST_TARGET}" in
    accuracy-test | accuracy-test-fast) ;;
    *)
        echo "ERROR: unsupported accuracy-test target: ${AIC_ACCURACY_TEST_TARGET}" >&2
        exit 2
        ;;
esac
SHORT="${SHA:0:7}"
AIC_IMAGE_NAME="rocm-aic-ci-${SHORT}"
AIC_SPUR_HOST="${AIC_SPUR_HOST:?AIC_SPUR_HOST must be set (e.g. via GitHub repo variable)}"
AIC_SPUR_HOST="${AIC_SPUR_HOST//[$'\t\r\n ']}"
AIC_SHARED_NFS="${AIC_SHARED_NFS:?AIC_SHARED_NFS must be set (e.g. via GitHub repo variable)}"
AIC_SPUR_CONTROLLER="${AIC_SPUR_CONTROLLER:?AIC_SPUR_CONTROLLER must be set (e.g. via GitHub repo variable)}"
AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT:-}"
KEEP_ARTIFACTS="${KEEP_ARTIFACTS:-0}"
REPO="https://github.com/ROCm/rocm-aic.git"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.github/scripts/runners/spur-ci-common.sh
source "${SCRIPT_DIR}/spur-ci-common.sh"
aic_ci_session_init "${SHORT}" "accuracy-test"

# Where the remote archives its per-job logs, and where they land here.  When
# AIC_CI_STORAGE_ROOT is unset this stays relative, which both the remote's
# mkdir (its shell starts in $HOME) and scp below resolve against the remote
# home — so the two halves agree without the runner knowing the remote $HOME.
# The run key is in the filename because a re-run, or a second workflow at the
# same SHA, would otherwise overwrite the archive the first one is still
# fetching.
ACCURACY_LOG_ARCHIVE_DIR="${AIC_CI_STORAGE_ROOT:+${AIC_CI_STORAGE_ROOT}/}accuracy-logs"
ACCURACY_LOG_ARCHIVE="${ACCURACY_LOG_ARCHIVE_DIR}/accuracy-${SHORT}.${AIC_CI_RUN_KEY}.tar.gz"
AIC_ACCURACY_LOG_DEST="${AIC_ACCURACY_LOG_DEST:-${RUNNER_TEMP:-/tmp}/accuracy-logs}"

# Read one field out of a JSON object the gate itself wrote (one line, no
# nesting), so a regex is sufficient and python3 need not exist on the runner.
# Prints nothing when the file or the field is absent.
_json_field() {
    local file="$1" key="$2"
    [[ -r "${file}" ]] || return 0
    awk -v key="${key}" '
        match($0, "\"" key "\"[[:space:]]*:[[:space:]]*(\"[^\"]*\"|[-+0-9.eE]+)") {
            field = substr($0, RSTART, RLENGTH)
            sub(/^[^:]*:[[:space:]]*/, "", field)
            gsub(/"/, "", field)
            print field
            exit
        }' "${file}"
}

# 4 dp, matching what the tests print and still far finer than the tolerances
# they gate on.  Empty in, empty out, so an absent score stays absent.
_fmt_score() {
    awk 'NF { printf "%.4f", $1 }'
}

# Report the scores the gate measured.  Best-effort from end to end: a missing
# or unreadable score thins the report, it never reddens a gate that passed and
# never hides one that failed -- the verdict is the exit code below, not this.
_report_scores() {
    local archive tmp arm file model delta
    archive="${AIC_ACCURACY_LOG_DEST}/${ACCURACY_LOG_ARCHIVE##*/}"
    [[ -r "${archive}" ]] || return 0
    tmp="$(mktemp -d)" || return 0
    if ! tar -xzf "${archive}" -C "${tmp}"; then
        echo "WARNING: could not unpack ${archive} to read the measured scores" >&2
        rm -rf "${tmp}"
        return 0
    fi

    # baseline-score.json is written by phase 1, tiered-score.json by phase 2 and
    # restart-score.json by phase 4, all under logs/<slurm-job-id>/.  A gate that
    # died early leaves the later ones absent, which is reported as absent.
    local -A scores=()
    local found=0
    shopt -s nullglob
    for arm in baseline tiered restart; do
        for file in "${tmp}"/logs/*/"${arm}"-score.json; do
            scores["${arm}"]="$(_json_field "${file}" score | _fmt_score)"
            [[ -z "${model:-}" ]] && model="$(_json_field "${file}" model)"
        done
        [[ -n "${scores[${arm}]:-}" ]] && found=1
    done
    shopt -u nullglob

    if (( found == 0 )); then
        echo "WARNING: ${archive} recorded no scores" >&2
        rm -rf "${tmp}"
        return 0
    fi

    # Printed, not asserted on: the tolerance lives in tests/accuracy and is
    # already enforced there, and duplicating it here would let the two drift.
    delta=""
    if [[ -n "${scores[baseline]:-}" && -n "${scores[tiered]:-}" ]]; then
        delta="$(awk -v t="${scores[tiered]}" -v b="${scores[baseline]}" \
            'BEGIN { printf "%+.4f", t - b }')"
    fi

    {
        echo "### Accuracy gate — measured gsm8k scores"
        echo
        echo "\`${model:-unknown model}\`, exact_match strict-match, 5-shot, full 1319-item split."
        echo
        echo "| arm | score |"
        echo "|---|---|"
        [[ -n "${scores[baseline]:-}" ]] && echo "| baseline (VRAM-only) | ${scores[baseline]} |"
        [[ -n "${scores[tiered]:-}" ]] && echo "| tiered (DRAM/NVMe) | ${scores[tiered]} |"
        [[ -n "${scores[restart]:-}" ]] && echo "| tiered, after vLLM restart | ${scores[restart]} |"
        [[ -n "${delta}" ]] && echo "| Δ tiered − baseline | ${delta} |"
        echo
    } | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}" || true

    # Consumed by the PR comment in aic-amd-dist-build-fast.yml.  The runner
    # collects this file whether or not the step succeeded, so the comment on a
    # failed gate still carries whichever numbers were reached.
    if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
        {
            echo "model=${model:-}"
            echo "baseline=${scores[baseline]:-}"
            echo "tiered=${scores[tiered]:-}"
            echo "restart=${scores[restart]:-}"
            echo "delta=${delta}"
        } >> "${GITHUB_OUTPUT}" || echo "WARNING: could not write step outputs" >&2
    fi

    rm -rf "${tmp}"
}

rc=0
aic_ci_ssh_bash \
    ACCURACY_LOG_ARCHIVE_DIR="${ACCURACY_LOG_ARCHIVE_DIR}" \
    ACCURACY_LOG_ARCHIVE="${ACCURACY_LOG_ARCHIVE}" \
    SHA="${SHA}" \
    REPO="${REPO}" \
    AIC_IMAGE_NAME="${AIC_IMAGE_NAME}" \
    AIC_ACCURACY_TEST_TARGET="${AIC_ACCURACY_TEST_TARGET}" \
    AIC_SHARED_NFS="${AIC_SHARED_NFS}" \
    AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT}" \
    KEEP_ARTIFACTS="${KEEP_ARTIFACTS}" \
    AIC_SPUR_CONTROLLER="${AIC_SPUR_CONTROLLER}" \
    SPUR_CONTROLLER_ADDR="${AIC_SPUR_CONTROLLER}" \
    HF_TOKEN="${HF_TOKEN:-}" << 'REMOTE' || rc=$?
set -euo pipefail

SHORT="${SHA:0:7}"
WORKDIR="$HOME/Projects/rocm-aic.${SHORT}"
CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT:-$HOME/Projects/rocm-aic-ci}"
TARBALL_DIR="${CI_STORAGE_ROOT}/images/aic-ci-${SHORT}"
CONTROL_PREFIX="${CI_STORAGE_ROOT}/control/${SHORT}.${AIC_CI_RUN_KEY}.${AIC_CI_STAGE}"
PID_FILE="${CONTROL_PREFIX}.pid"
JOB_FILE="${CONTROL_PREFIX}.job"
CANCEL_FILE="${CONTROL_PREFIX}.cancel"

mkdir -p "${CI_STORAGE_ROOT}/control"
printf '%s\n' "${BASHPID}" > "${PID_FILE}"
if [[ -e "${CANCEL_FILE}" ]]; then
    echo "CI session was cancelled before remote startup completed" >&2
    rm -f "${PID_FILE}" "${JOB_FILE}" "${CANCEL_FILE}" 2>/dev/null || true
    exit 143
fi
# The accuracy job holds a GPU for up to two hours, so an abandoned one is the
# most expensive orphan in the chain: record its id where the wrapper's cancel
# path can find it.
export AIC_CI_ACTIVE_JOB_FILE="${JOB_FILE}"

_best_effort_remove() {
    rm -rf "$@" || echo "WARNING: cleanup could not fully remove: $*" >&2
}

_cleanup() {
    echo "=== Cleaning up ==="
    # Keep logs/: _harvest_logs has already archived them outside WORKDIR, but a
    # failed archive should not also destroy the originals.  Mirrors
    # spur-smoke-test.sh, which preserves logs/ for the same reason.
    find "${WORKDIR}" -mindepth 1 -maxdepth 1 -not -name logs -exec rm -rf {} + ||
        echo "WARNING: cleanup could not fully remove the contents of ${WORKDIR}" >&2
    _best_effort_remove "${TARBALL_DIR}"
}

# The per-job accuracy logs (scores, per-arm container logs) live under
# WORKDIR/logs/$SLURM_JOB_ID and were previously destroyed by the cleanup above,
# leaving a failed gate with nothing but the Actions stdout tail.  Archive them
# outside WORKDIR before anything deletes it; the runner scp's this back and
# uploads it.  Runs on both paths — the nightly's downstream cliff harvest
# removes WORKDIR too.
#
# SPUR writes a job's stderr to spur-<jobid>.out beside the submitting shell
# rather than folding it into --output, so those files are archived alongside
# logs/: when a job dies before its own redirect is in place, they are the only
# record of why.
_harvest_logs() {
    echo "=== Archiving accuracy logs ==="
    if [[ ! -d "${WORKDIR}/logs" ]]; then
        echo "WARNING: no ${WORKDIR}/logs to archive" >&2
        return 0
    fi
    local -a members=(logs) f
    shopt -s nullglob
    for f in "${WORKDIR}"/spur-*.out; do members+=("${f#"${WORKDIR}"/}"); done
    shopt -u nullglob
    mkdir -p "${ACCURACY_LOG_ARCHIVE_DIR}" || {
        echo "WARNING: could not create ${ACCURACY_LOG_ARCHIVE_DIR}" >&2; return 0; }
    if tar -czf "${ACCURACY_LOG_ARCHIVE}" -C "${WORKDIR}" "${members[@]}"; then
        echo "=== Archived to $(hostname):${ACCURACY_LOG_ARCHIVE} ==="
    else
        echo "WARNING: failed to archive accuracy logs from ${WORKDIR}/logs" >&2
    fi
    # Bounded retention -- this is shared, quota'd NFS.  A failure here silently
    # leaks quota until it breaks unrelated jobs, so report it; do not let it
    # fail this path, which may already be handling an earlier failure.
    find "${ACCURACY_LOG_ARCHIVE_DIR}" -maxdepth 1 -name 'accuracy-*.tar.gz' -mtime +7 \
        -delete || echo "WARNING: could not prune old accuracy log archives" >&2
    return 0
}

_on_exit() {
    local rc=$?
    trap - EXIT
    _harvest_logs
    # KEEP_ARTIFACTS=1 means the downstream stage runs regardless of how this one
    # ends and reuses the artifacts, so removing them here — even on failure —
    # would break it.
    if [[ "${KEEP_ARTIFACTS}" == "1" ]]; then
        echo "=== KEEP_ARTIFACTS=1: downstream stage owns cleanup ==="
    else
        _cleanup
    fi
    if (( rc == 0 )); then
        rm -f "${PID_FILE}" "${JOB_FILE}" "${CANCEL_FILE}" 2>/dev/null || true
    fi
    exit "${rc}"
}
trap _on_exit EXIT

# Re-clone if WORKDIR is missing or checked out at the wrong SHA (e.g. stale
# leftover from a prior failed run at a different commit with the same prefix).
ACTUAL_SHA="$(git -C "${WORKDIR}" rev-parse HEAD 2>/dev/null || true)"
if [[ ! -d "${WORKDIR}" || "${ACTUAL_SHA}" != "${SHA}" ]]; then
    echo "=== (Re-)cloning ${REPO} at ${SHA} ==="
    rm -rf "${WORKDIR}"
    git clone --filter=blob:none --no-single-branch "${REPO}" "${WORKDIR}"
    git -C "${WORKDIR}" checkout "${SHA}"
fi

if [[ ! -d "${TARBALL_DIR}" ]]; then
    echo "ERROR: ${TARBALL_DIR} not found — did dist-build run first?" >&2
    exit 1
fi

echo "=== Running ${AIC_ACCURACY_TEST_TARGET} (AIC_IMAGE_NAME=${AIC_IMAGE_NAME}) ==="
AIC_SPUR_CLUSTER=1 \
    AIC_IMAGE_NAME="${AIC_IMAGE_NAME}" \
    AIC_IMAGE_DIR="${TARBALL_DIR}" \
    HF_TOKEN="${HF_TOKEN:-}" \
    make -C "${WORKDIR}" "${AIC_ACCURACY_TEST_TARGET}"

echo "=== ${AIC_ACCURACY_TEST_TARGET} complete ==="
REMOTE

# Fetch the archive whatever the gate concluded — a red gate is precisely when
# these logs are the only post-mortem.  `set -e` would have skipped this on
# failure, hence the explicit rc capture above.
mkdir -p "${AIC_ACCURACY_LOG_DEST}"
if scp -q "${AIC_SPUR_HOST}:${ACCURACY_LOG_ARCHIVE}" "${AIC_ACCURACY_LOG_DEST}/"; then
    echo "Accuracy logs retrieved to ${AIC_ACCURACY_LOG_DEST}/$(basename "${ACCURACY_LOG_ARCHIVE}")"
else
    echo "WARNING: could not retrieve accuracy logs for ${SHORT}" >&2
fi

_report_scores

if [[ "${rc}" -ne 0 ]]; then
    echo "Accuracy test FAILED for ${SHORT} (exit ${rc})" >&2
    exit "${rc}"
fi

echo "Accuracy test passed for ${SHORT}"
