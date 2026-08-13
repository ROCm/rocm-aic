#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Overnight matrix driver.  Submits .slurm/overnight-tp.sbatch once per
# configuration, STRICTLY SERIALLY -- the compose stack uses fixed container
# names and grafana binds host port 3000, so two runs on the same node collide.
#
# Results are appended to overnight.md as each run finishes, so the file is
# useful even if the driver is interrupted half way through.
#
#   nohup .slurm/overnight-run.sh > logs/overnight-driver.log 2>&1 &
set -uo pipefail

AIC_DAY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${AIC_DAY_DIR}" || exit 1
# shellcheck source=/dev/null
[ -f .env.aic ] && source .env.aic >/dev/null 2>&1

MD="${AIC_DAY_DIR}/overnight.md"
JOB="${AIC_DAY_DIR}/.slurm/overnight-tp.sbatch"
PART="${AIC_TEST_PARTITION:-storage}"
ACCT="${AIC_TEST_ACCOUNT:-gds}"
ARCH="$(bash "${AIC_DAY_DIR}/.slurm/aic-test-arch.sh" 2>/dev/null || echo gfx942)"
TAG="$(bash "${AIC_DAY_DIR}/docker/scripts/aic-image-tag.sh" 2>/dev/null)"
IMAGE="${AIC_IMAGE:-rocm-aic:${TAG}}"
DIR="${AIC_IMAGE_DIR:-/scratch/${USER}/images}"
TARBALL="${DIR}/rocm-aic-${TAG}-${ARCH}.tar.zst"

# Stop before the user is back rather than leaving a run half-done.
DEADLINE="${OVERNIGHT_DEADLINE:-$(date -d 'tomorrow 07:30' +%s)}"

log() { printf '[driver %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
md()  { printf '%s\n' "$*" >> "${MD}"; }

[ -r "${TARBALL}" ] || { log "FATAL: no tarball ${TARBALL}"; exit 1; }

# --- the matrix -------------------------------------------------------------
# name|TP|DEVS|MODEL|L2|L1_GB|GPU_UTIL|CONC
# Ordered cheapest-and-most-informative first: if the night is cut short we
# still have the TP scaling curve, which is the headline question.
CONFIGS=(
  "tp1-baseline|1|0|Qwen/Qwen2.5-3B-Instruct|none|8|0.85|16"
  "tp2|2|0,1|Qwen/Qwen2.5-3B-Instruct|none|8|0.85|16"
  "tp4|4|0,1,2,3|Qwen/Qwen2.5-3B-Instruct|none|8|0.85|16"
  "tp8|8|0,1,2,3,4,5,6,7|Qwen/Qwen2.5-3B-Instruct|none|8|0.85|16"
  "tp4-conc64|4|0,1,2,3|Qwen/Qwen2.5-3B-Instruct|none|8|0.85|64"
  "tp1-conc64|1|0|Qwen/Qwen2.5-3B-Instruct|none|8|0.85|64"
  "tp4-l2posix|4|0,1,2,3|Qwen/Qwen2.5-3B-Instruct|nixl_posix|1|0.85|16"
  "tp1-l2posix|1|0|Qwen/Qwen2.5-3B-Instruct|nixl_posix|1|0.85|16"
  "tp4-l2localdisk|4|0,1,2,3|Qwen/Qwen2.5-3B-Instruct|local_disk|1|0.85|16"
  "tp4-util50|4|0,1,2,3|Qwen/Qwen2.5-3B-Instruct|none|8|0.50|16"
  "tp8-conc64|8|0,1,2,3,4,5,6,7|Qwen/Qwen2.5-3B-Instruct|none|8|0.85|64"
  "tp1-tiny|1|0|Qwen/Qwen2.5-0.5B-Instruct|none|8|0.85|16"
)

if [ ! -s "${MD}" ]; then
  md "# Overnight run — $(date '+%Y-%m-%d %H:%M')"
  md ""
  md "Node \`ctr-smc-mi300x-cx68-25\` (8x MI300X, gfx942), partition \`${PART}\`, account \`${ACCT}\`."
  md "Image \`${IMAGE}\`. Driver: \`.slurm/overnight-run.sh\`, job: \`.slurm/overnight-tp.sbatch\`."
  md ""
  md "Runs are strictly serial: the compose stack uses fixed container names and"
  md "grafana binds host port 3000, so two concurrent runs on one node collide."
  md ""
  md "## Results"
  md ""
  md "| run | TP | devices | L2 | L1 GB | util | conc | ready s | tok/s | status |"
  md "|-----|----|---------|----|-------|------|------|---------|-------|--------|"
fi

log "matrix: ${#CONFIGS[@]} configs, deadline $(date -d "@${DEADLINE}" '+%a %H:%M')"

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name tp devs model l2 l1 util conc <<< "${cfg}"
  now=$(date +%s)
  if [ "${now}" -ge "${DEADLINE}" ]; then
    log "deadline reached; stopping before ${name}"
    md "| ${name} | ${tp} | ${devs} | ${l2} | ${l1} | ${util} | ${conc} | - | - | skipped (deadline) |"
    continue
  fi

  log "submitting ${name} (tp=${tp} devs=${devs} l2=${l2} conc=${conc})"
  # Export into the environment and use a bare --export=ALL.  Do NOT inline
  # VAR=value pairs after --export=ALL: that list is COMMA-separated, so a value
  # containing a comma (DEVS="0,1,2,3") is truncated at the first comma.  That
  # silently handed every TP>1 run a single GPU and made them all fail.
  export AIC_DAY_DIR AIC_IMAGE="${IMAGE}" TARBALL RUN_NAME="${name}" \
         TP="${tp}" DEVS="${devs}" MODEL="${model}" L2="${l2}" \
         L1_GB="${l1}" GPU_UTIL="${util}" CONC="${conc}"
  out="$(sbatch --parsable --wait \
      -A "${ACCT}" -p "${PART}" \
      --job-name="ovn-${name}" \
      --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=128G \
      --gres=gpu:8 --time=01:00:00 \
      --output="${AIC_DAY_DIR}/logs/overnight-${name}.out" \
      --export=ALL \
      "${JOB}" 2>&1)"
  rc=$?
  jid="$(printf '%s' "${out}" | grep -oE '^[0-9]+' | head -1)"
  logf="${AIC_DAY_DIR}/logs/overnight-${name}.out"
  log "${name}: job=${jid:-?} rc=${rc}"

  line="$(grep -h '^RESULT:' "${logf}" 2>/dev/null | tail -1)"
  if [ -n "${line}" ]; then
    get() { printf '%s' "${line}" | grep -oE "$1=[^ ]*" | cut -d= -f2-; }
    st="$(get status)"
    md "| ${name} | ${tp} | \`${devs}\` | ${l2} | ${l1} | ${util} | ${conc} | $(get ttr_s) | **$(get tok_s)** | ${st} |"
  else
    st="no-result"
    md "| ${name} | ${tp} | \`${devs}\` | ${l2} | ${l1} | ${util} | ${conc} | - | - | no RESULT (job ${jid:-?}, rc=${rc}) |"
  fi

  # The baseline is the known-good config (TP=1 is what we have already proven
  # end-to-end).  If IT fails the harness is broken, not the configuration, and
  # repeating it 11 more times just burns the night.
  if [ "${name}" = "tp1-baseline" ] && [ "${st}" != "ok" ]; then
    log "ABORT: baseline failed (${st}) -- harness problem, not a config problem"
    md ""
    md "> **Aborted after the baseline failed (\`${st}\`).** TP=1 is the configuration"
    md "> already proven end-to-end today, so a failure there means the overnight"
    md "> harness is broken rather than the setting under test. See"
    md "> \`logs/overnight-tp1-baseline.out\`. No further configs were run."
    exit 1
  fi
done

md ""
md "_Driver finished $(date '+%Y-%m-%d %H:%M')._"
log "matrix complete"
