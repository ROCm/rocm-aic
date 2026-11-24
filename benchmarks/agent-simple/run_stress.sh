#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Run stress.py against the running AIC stack from ${AIC_STACK}-client, the
# client container the monitoring profile starts on the stack's docker network.
#
#   run_stress.sh <run-id> <documents> <concurrency> <seconds> [<max-words>]
#   run_stress.sh s1 220 16 480
#
# AIC_BENCH_ENDPOINT: vLLM base URL on the aic bridge (default
#   http://aic-lmcache:8000).  The LMCache and NIXL exporters are taken from
#   the same host; see benchmarks/common/kv_client.py for their overrides.
set -euo pipefail

# AIC_STACK, and so the client's container name.
# shellcheck source=scripts/aic-endpoints.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/aic-endpoints.sh"
client="${AIC_STACK}-client"

[[ "$(docker inspect -f '{{.State.Running}}' "${client}" 2>/dev/null)" == true ]] || {
    echo "ERROR: ${client} container not running; start the stack with 'make up'" >&2
    exit 1
}
env_args=()
for var in AIC_BENCH_ENDPOINT AIC_BENCH_MODEL AIC_BENCH_LMCACHE_METRICS_URL AIC_BENCH_NIXL_METRICS_URL; do
    [[ -z "${!var:-}" ]] || env_args+=(-e "${var}")
done
exec docker exec "${env_args[@]}" "${client}" aic-bench agent-simple "$@"
