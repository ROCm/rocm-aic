#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Clone NIXL for ROCm recipe images (andyluo7/nixl amd-support + overlays).
set -euo pipefail

: "${NIXL_GIT_URL:?NIXL_GIT_URL is required}"
NIXL_DEST="${NIXL_DEST:-/tmp/nixl}"

NIXL_SHA="${NIXL_SHA:-${NIXL_AMD_SUPPORT_SHA:-}}"
NIXL_REF="${NIXL_REF:-amd-support}"

if [[ "${NIXL_GIT_URL}" == git@* ]]; then
	echo "ERROR: git@ NIXL_GIT_URL is not supported in this Dockerfile layer." >&2
	echo "       Use https://github.com/andyluo7/nixl.git (public) or an HTTPS mirror." >&2
	exit 1
fi

rm -rf "${NIXL_DEST}"

if [[ -n "${NIXL_SHA}" ]]; then
	git init "${NIXL_DEST}"
	git -C "${NIXL_DEST}" remote add origin "${NIXL_GIT_URL}"
	if ! git -C "${NIXL_DEST}" fetch --depth 1 origin "${NIXL_SHA}"; then
		echo "ERROR: failed to fetch NIXL_SHA=${NIXL_SHA} from ${NIXL_GIT_URL}" >&2
		exit 1
	fi
	git -C "${NIXL_DEST}" checkout FETCH_HEAD
	echo "NIXL cloned to ${NIXL_DEST} sha=${NIXL_SHA}"
elif [[ -n "${NIXL_REF}" ]]; then
	if ! git clone --depth 1 --branch "${NIXL_REF}" "${NIXL_GIT_URL}" "${NIXL_DEST}"; then
		echo "ERROR: failed to clone branch ${NIXL_REF} from ${NIXL_GIT_URL}" >&2
		exit 1
	fi
	echo "NIXL cloned to ${NIXL_DEST} ref=${NIXL_REF} sha=$(git -C "${NIXL_DEST}" rev-parse HEAD)"
else
	git clone --depth 1 "${NIXL_GIT_URL}" "${NIXL_DEST}"
	echo "NIXL cloned to ${NIXL_DEST} ref=default branch"
fi

if [[ "${NIXL_REQUIRE_ROCM:-0}" == "1" ]]; then
	if [[ ! -f "${NIXL_DEST}/meson_options.txt" ]] \
		|| ! grep -q "option('wheel_variant'" "${NIXL_DEST}/meson_options.txt"; then
		echo "ERROR: ${NIXL_DEST} lacks amd-support wheel_variant meson option." >&2
		echo "       Set NIXL_SHA or NIXL_REF=amd-support in defaults.mk." >&2
		exit 1
	fi
fi
