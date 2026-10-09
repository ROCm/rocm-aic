#!/usr/bin/env bash
# Reads docker buildx --progress=plain output from stdin and emits timestamped
# progress lines.  Pipe make build output through this script:
#
#   make build ... 2>&1 | tee /tmp/build-raw.log | scripts/time-docker-build.sh
#
# Output goes to /tmp/build-timed.log and stdout.
set -euo pipefail

LOG=/tmp/build-timed.log
: > "$LOG"

t0=$(date +%s)
step_t=$t0

emit() {
    local now total step_elapsed msg
    now=$(date +%s)
    total=$(( now - t0 ))
    step_elapsed=$(( now - step_t ))
    msg="[$(date +%H:%M:%S) | total=${total}s step=${step_elapsed}s]   $1"
    echo "$msg" | tee -a "$LOG"
    step_t=$now
}

emit_stage() {
    step_t=$(date +%s)
    emit "STAGE $1"
}

stage=0
current_step=""

while IFS= read -r line; do
    # Stage banner from make build: "--- build [N/M]: image:tag ..."
    if [[ "$line" =~ ^---\ build\ \[([0-9]+/[0-9]+)\]:\ (.+) ]]; then
        stage=$(( stage + 1 ))
        emit_stage "${BASH_REMATCH[1]}: ${BASH_REMATCH[2]}"
        continue
    fi

    # BuildKit step header: "#NN [stage step/total] description"
    if [[ "$line" =~ ^#([0-9]+)\ \[([^]]+)\]\ (.+) ]]; then
        step="${BASH_REMATCH[1]}"
        loc="${BASH_REMATCH[2]}"
        desc="${BASH_REMATCH[3]}"
        if [[ "$step $loc" != "$current_step" ]]; then
            current_step="$step $loc"
            emit "#${step} [${loc}] ${desc}"
        fi
        continue
    fi

    # Ninja progress: "#NN ninja NNN/TOTAL (PCT%)"
    if [[ "$line" =~ ^#([0-9]+)\ ([0-9]+)/([0-9]+)\ \(([0-9]+)%\) ]]; then
        step="${BASH_REMATCH[1]}"
        done="${BASH_REMATCH[2]}"
        total="${BASH_REMATCH[3]}"
        pct="${BASH_REMATCH[4]}"
        # emit at every 10% boundary
        bucket=$(( pct / 10 ))
        key="#${step} ninja ${bucket}"
        if [[ "$key" != "$current_step" ]]; then
            current_step="$key"
            emit "#${step} ninja ${done}/${total} (${pct}%)"
        fi
        continue
    fi
done

emit "BUILD COMPLETE"
