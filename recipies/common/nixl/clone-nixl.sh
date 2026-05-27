#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Clone NIXL for ROCm recipe images. Default: andyluo7/nixl amd-support branch.
# git@ URLs are rejected; use HTTPS mirrors for private forks.
set -euo pipefail

: "${NIXL_GIT_URL:?NIXL_GIT_URL is required}"
NIXL_REF="${NIXL_REF:-amd-support}"
NIXL_DEST="${NIXL_DEST:-/tmp/nixl}"

if [[ "${NIXL_GIT_URL}" == git@* ]]; then
	echo "ERROR: git@ NIXL_GIT_URL is not supported in this Dockerfile layer." >&2
	echo "       Use https://github.com/andyluo7/nixl.git (public) or an HTTPS mirror." >&2
	exit 1
fi

rm -rf "${NIXL_DEST}"

if [[ -n "${NIXL_REF}" ]]; then
	if git clone --depth 1 --branch "${NIXL_REF}" "${NIXL_GIT_URL}" "${NIXL_DEST}"; then
		:
	else
		rm -rf "${NIXL_DEST}"
		git clone "${NIXL_GIT_URL}" "${NIXL_DEST}"
		git -C "${NIXL_DEST}" checkout "${NIXL_REF}"
	fi
else
	git clone --depth 1 "${NIXL_GIT_URL}" "${NIXL_DEST}"
fi

# Optional pin when NIXL_REF is a branch name (defaults.mk NIXL_AMD_SUPPORT_SHA).
if [[ -n "${NIXL_AMD_SUPPORT_SHA:-}" && "${NIXL_REF:-}" == amd-support ]]; then
	git -C "${NIXL_DEST}" fetch --depth 1 origin "${NIXL_AMD_SUPPORT_SHA}" 2>/dev/null || true
	git -C "${NIXL_DEST}" checkout "${NIXL_AMD_SUPPORT_SHA}" 2>/dev/null || true
fi

echo "NIXL cloned to ${NIXL_DEST} ref=${NIXL_REF:-HEAD}"
