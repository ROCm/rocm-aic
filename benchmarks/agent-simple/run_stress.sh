#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Run stress.py against the running AIC stack from aic-client, the client
# container the monitoring profile starts on the stack's docker network with the
# repo mounted at /repo.
#
#   run_stress.sh <run-id> <documents> <concurrency> <seconds> [<max-words>]
#   run_stress.sh s1 220 16 480
#
# AIC_STRESS_HOST: container owning vLLM's network namespace (default aic-lmcache).
set -euo pipefail

docker inspect aic-client >/dev/null 2>&1 || {
    echo "ERROR: aic-client container not running; start the stack with 'make up'" >&2
    exit 1
}
exec docker exec -e AIC_STRESS_HOST="${AIC_STRESS_HOST:-aic-lmcache}" \
    aic-client python3 -u benchmarks/agent-simple/stress.py "$@"
