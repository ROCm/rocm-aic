#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Clone ROCm/nixl for the inference image. Default path is HTTPS (no BuildKit
# SSH). git@ URLs are rejected here; use an HTTPS mirror or extend the
# Dockerfile with an SSH-mounted clone if you must build from a private SSH
# remote.
set -euo pipefail

: "${NIXL_GIT_URL:?NIXL_GIT_URL is required}"
NIXL_REF="${NIXL_REF:-}"

if [[ "${NIXL_GIT_URL}" == git@* ]]; then
	echo "ERROR: git@ NIXL_GIT_URL is not supported in this Dockerfile layer." >&2
	echo "       Use https://github.com/ROCm/nixl.git (public) or an HTTPS mirror." >&2
	exit 1
fi

rm -rf /tmp/nixl

if [[ -n "${NIXL_REF}" ]]; then
	if git clone --depth 1 --branch "${NIXL_REF}" "${NIXL_GIT_URL}" /tmp/nixl; then
		:
	else
		rm -rf /tmp/nixl
		git clone "${NIXL_GIT_URL}" /tmp/nixl
		git -C /tmp/nixl checkout "${NIXL_REF}"
	fi
else
	git clone --depth 1 "${NIXL_GIT_URL}" /tmp/nixl
fi
