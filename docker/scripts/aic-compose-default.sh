#!/usr/bin/env bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Print the default docker/docker-compose.yml gives VAR: the text after `:-` in
# its first `${VAR:-default}`.  Compose owns the runtime image pins, and a few
# callers need one before the stack is up, on hosts that may not have the
# compose plugin (image pre-pulls, `make dist-build-monitoring`).  They read it
# here rather than keep a copy.  Plain defaults only: a nested `${A:-${B:-x}}`
# is not understood.
#
# Usage: aic-compose-default.sh VAR
set -euo pipefail

var="${1:?usage: aic-compose-default.sh VAR}"
[[ "${var}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || {
    echo "aic-compose-default.sh: not a variable name: ${var}" >&2
    exit 2
}
compose="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/docker-compose.yml"
val="$(sed -n "/.*\${${var}:-\([^\${}]*\)}.*/{s//\1/p;q}" "${compose}")"
[[ -n "${val}" ]] || {
    echo "aic-compose-default.sh: no plain \${${var}:-...} default in ${compose}" >&2
    exit 1
}
printf '%s\n' "${val}"
