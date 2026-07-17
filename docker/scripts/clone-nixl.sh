#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Clone NIXL from the ai-dynamo upstream at a pinned ref for AIC image builds.
set -euo pipefail

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
