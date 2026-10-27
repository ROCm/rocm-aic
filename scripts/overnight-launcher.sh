#!/usr/bin/env bash
# Wait for model download to complete, then launch overnight-test.sh.
# This runs detached so the user can leave.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${REPO_ROOT}/logs/overnight-2026-08-12"
LOG="${LOGDIR}/launcher.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG}"; }

log "Launcher started. Waiting for model download to complete..."

# Wait for all snapshot_download processes to finish
while pgrep -f "snapshot_download" > /dev/null 2>&1; do
    COUNT=$(find ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B -name "*.safetensors" 2>/dev/null | wc -l || echo 0)
    INCOMPLETE=$(find ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/blobs -name "*.incomplete" 2>/dev/null | wc -l || echo 0)
    log "  waiting... safetensors=${COUNT} incomplete=${INCOMPLETE}"
    sleep 30
done

WEIGHT_COUNT=$(find ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B -name "*.safetensors" 2>/dev/null | wc -l || echo 0)
log "Download complete: ${WEIGHT_COUNT} safetensors files"

if [ "${WEIGHT_COUNT}" -lt 3 ]; then
    log "ERROR: Too few weight files (${WEIGHT_COUNT}). Model download may have failed."
    log "Check: ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/"
    exit 1
fi

# Clean up any leftover containers from previous attempts
docker rm -f aic-lmcache-test aic-vllm-test aic-prometheus-overnight 2>/dev/null || true
docker network rm aic-test 2>/dev/null || true
sleep 3

log "Launching overnight-test.sh..."
mv "${LOGDIR}/overnight-master.log" "${LOGDIR}/overnight-master.log.prev3" 2>/dev/null || true
exec bash "${REPO_ROOT}/scripts/overnight-test.sh" \
    >> "${LOGDIR}/overnight-master.log" 2>&1
