#!/usr/bin/env bash
# Validate ONE dual-arch image on EACH of the architectures it was built for.
#
# The image stays constant (AIC_ROCM_ARCH='gfx942;gfx950', tag ...-gfx942-950);
# only AIC_TEST_CONSTRAINT changes, pinning each pass to one GPU family.  That
# is the whole point: a dual-arch image is only as good as its worst target,
# and the previous run proved nothing because Slurm placed every job on a
# gfx1151 APU the image never covered.
#
# Sequential throughout: every test uses the same fixed container names
# (aic-vllm-gpu0, aic-lmcache) and vllm publishes a host port, so two at once
# on one node collide.
#
#   .slurm/overnight-dualarch.sh <report.md>
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT="${1:-${REPO}/logs/dualarch-$(date +%Y%m%d-%H%M).md}"

export ROCM_ARCH='gfx942;gfx950'
export AIC_ROCM_ARCH='gfx942;gfx950'
IMAGE_ARCHS="gfx942 gfx950"
TARGETS="smoke-test tiny-test accuracy-test"

mkdir -p "$(dirname "${REPORT}")"
say() { printf '%s\n' "$*" >> "${REPORT}"; }

say "# Dual-arch validation — $(date '+%Y-%m-%d %H:%M')"
say ""
say "One image (\`${AIC_ROCM_ARCH}\`) exercised on each architecture it targets."
say "Only \`AIC_TEST_CONSTRAINT\` varies between passes; the image is identical."
say ""
say "| arch | target | job | node | node gfx | state | exit | wall | TSDB |"
say "|---|---|---|---|---|---|---|---|---|"

declare -A RESULT
for arch in ${IMAGE_ARCHS}; do
    GFX="$(printf '%s' "${arch}" | tr '[:lower:]' '[:upper:]')"
    # Explicit constraint wins over the derived one, pinning this pass to one
    # GPU family while the image stays dual-arch.
    export AIC_TEST_CONSTRAINT="${AIC_TEST_SITE:-MARKHAM}&${GFX}"
    for target in ${TARGETS}; do
        logname="${target}"
        t0="$(date +%s)"
        out="$(cd "${REPO}" && make "${target}" 2>&1)" || true
        t1="$(date +%s)"
        jobid="$(printf '%s\n' "${out}" | grep -oE 'as job [0-9]+' | grep -oE '[0-9]+$' | tail -1)"
        wall="$(( (t1 - t0) / 60 ))m"
        if [[ -z "${jobid}" ]]; then
            say "| ${arch} | ${target} | — | — | — | submit-failed | — | ${wall} | — |"
            RESULT["${arch}/${target}"]="submit-failed"
            continue
        fi
        node="$(sacct -j "${jobid}" --format=NodeList -n 2>/dev/null | head -1 | tr -d ' ')"
        nodegfx="$(sinfo -N -n "${node}" -h -o '%f' 2>/dev/null | head -1 | grep -oE 'GFX[0-9A-Za-z]+' || true)"
        state="$(sacct -j "${jobid}" --format=State -n 2>/dev/null | head -1 | tr -d ' ')"
        # The exit file, not sacct: a requeued job reports PENDING for work
        # that already succeeded.
        exitf="$(cat "${REPO}/logs/${jobid}/${logname}.exit" 2>/dev/null || echo '?')"
        tsdb="$(du -sh "${REPO}/logs/${jobid}/prometheus" 2>/dev/null | cut -f1 || true)"
        say "| ${arch} | ${target} | \`${jobid}\` | ${node} | ${nodegfx:-?} | ${state} | \`${exitf}\` | ${wall} | ${tsdb:-—} |"
        RESULT["${arch}/${target}"]="${exitf}"
        # Stash the interesting lines for the detail section below.
        L="${REPO}/logs/${jobid}/${logname}.out"
        if [[ -s "${L}" ]]; then
            {
                printf '\n<!--DETAIL %s %s %s-->\n' "${arch}" "${target}" "${jobid}"
                sed 's/\x1b\[[0-9;]*m//g' "${L}" \
                  | grep -iE "ALL CHECKS PASSED|ALL PHASES PASSED|non-empty completion|OK Phase|FAIL|ERROR:|AcceleratorError|kernel image|Traceback|up=1 job=|score" \
                  | tail -20
            } >> "${REPORT}.detail"
        fi
    done
done

say ""
say "## Detail"
say ""
if [[ -s "${REPORT}.detail" ]]; then
    while IFS= read -r line; do
        case "${line}" in
            "<!--DETAIL "*)
                # shellcheck disable=SC2086  # deliberate split on the marker
                set -- ${line}
                say ""; say "### $2 / $3 (job $4)"; say '```' ;;
            "") : ;;
            *) say "${line}" ;;
        esac
    done < "${REPORT}.detail"
    say '```'
    rm -f "${REPORT}.detail"
fi

say ""
say "## Summary"
say ""
_pass=0; _fail=0
for k in "${!RESULT[@]}"; do
    [[ "${RESULT[$k]}" == "0" ]] && _pass=$((_pass+1)) || _fail=$((_fail+1))
done
say "- passed: ${_pass}"
say "- failed: ${_fail}"
say "- finished: $(date '+%Y-%m-%d %H:%M')"
