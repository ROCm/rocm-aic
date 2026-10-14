#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Clone NIXL from the ai-dynamo upstream at a pinned ref for AIC image builds.
# -E inherit ERR trap in functions and subshells,
# -e exit the script upon commands failing,
# -u to error if the script ever references a variable that is unset,
# -o pipefail to error if any stage of a pipeline fails.
set -Eeuo pipefail
# errexit exits without saying where; report it.  Only the line number and
# function call stack are printed, never the command text, which can carry
# secret values.
_on_err() {
	# set -E carries this trap into $(...), where errexit is off.
	[[ $- == *e* ]] || return 0
	# || : so a failed write cannot replace the exit status.
	printf '%s: command failed (exit %s, pipeline status %s) at line %s%s\n' \
		"${0##*/}" "$1" "$2" "$3" "${4:+ in ${4// / <- }}" >&2 || :
}
trap '_on_err "$?" "${PIPESTATUS[*]}" "$LINENO" "${FUNCNAME[*]-}"' ERR

: "${NIXL_GIT_URL:?NIXL_GIT_URL is required}"
: "${NIXL_REF:?NIXL_REF is required}"
NIXL_DEST="${NIXL_DEST:-/tmp/nixl}"

if [[ "${NIXL_GIT_URL}" == git@* ]]; then
	echo "ERROR: git@ NIXL_GIT_URL is not supported in Dockerfile layers; use HTTPS." >&2
	exit 1
fi

rm -rf "${NIXL_DEST}"

if ! git clone --depth 1 --branch "${NIXL_REF}" "${NIXL_GIT_URL}" "${NIXL_DEST}"; then
	echo "ERROR: failed to clone ${NIXL_GIT_URL} at ref ${NIXL_REF}" >&2
	exit 1
fi
echo "NIXL cloned to ${NIXL_DEST} ref=${NIXL_REF} sha=$(git -C "${NIXL_DEST}" rev-parse HEAD)"

if [[ "${NIXL_REQUIRE_ROCM:-0}" == "1" ]]; then
	if [[ ! -f "${NIXL_DEST}/meson_options.txt" ]] \
		|| ! grep -q "option('wheel_variant'" "${NIXL_DEST}/meson_options.txt"; then
		echo "ERROR: ${NIXL_DEST} lacks the wheel_variant meson option — wrong ref?" >&2
		exit 1
	fi
fi
