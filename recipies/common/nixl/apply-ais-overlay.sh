#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Apply AIS/AIS_MT hipfile plugins onto andyluo7/nixl amd-support checkout.
# Overlay sources live under recipies/common/nixl/overlay/.
set -euo pipefail

NIXL_SRC="${NIXL_SRC:-/tmp/nixl}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERLAY="${SCRIPT_DIR}/overlay"
PATCH="${SCRIPT_DIR}/patches/amd-support-ais.patch"

if [[ ! -f "${NIXL_SRC}/meson.build" ]]; then
	echo "ERROR: ${NIXL_SRC}/meson.build not found" >&2
	exit 1
fi

# Detect whether this nixl checkout already has native HIP + AIS_MT support
# (sbates130272/nixl@9d14642+ and later). If so, no patching or overlay copying
# is needed — the repo ships everything we need out of the box.
_NATIVE_AIS_MT=0
if grep -q "enabled_plugins.get('AIS_MT')" "${NIXL_SRC}/src/plugins/meson.build" 2>/dev/null \
	&& grep -q "hip_dep" "${NIXL_SRC}/meson.build" 2>/dev/null; then
	_NATIVE_AIS_MT=1
fi

if [[ "${_NATIVE_AIS_MT}" == "1" ]]; then
	echo "nixl has native HIP + AIS_MT support; skipping overlay and meson patches"
	echo "AIS overlay complete for ${NIXL_SRC}"
	exit 0
fi

# --- Older nixl (andyluo7/nixl amd-support pre-native-HIP) ---

# Inject ROCm/HIP toolchain support into meson.build.
python3 "${SCRIPT_DIR}/patch-rocm-meson.py"

if [[ -d "${OVERLAY}/src/plugins/ais" ]]; then
	cp -r "${OVERLAY}/src/plugins/ais" "${NIXL_SRC}/src/plugins/"
	cp -r "${OVERLAY}/src/plugins/ais_mt" "${NIXL_SRC}/src/plugins/"
	echo "Copied AIS and AIS_MT plugin sources"
fi

if [[ -d "${OVERLAY}/test/unit/plugins/ais_mt" ]]; then
	mkdir -p "${NIXL_SRC}/test/unit/plugins"
	cp -r "${OVERLAY}/test/unit/plugins/ais_mt" "${NIXL_SRC}/test/unit/plugins/"
	echo "Copied AIS_MT unit tests"
fi

if [[ -f "${PATCH}" ]] && git -C "${NIXL_SRC}" apply --check "${PATCH}" 2>/dev/null; then
	git -C "${NIXL_SRC}" apply "${PATCH}"
	echo "Applied ${PATCH}"
elif grep -q "'AIS'" "${NIXL_SRC}/meson.build" 2>/dev/null; then
	echo "AIS already present in meson.build; skipping patch"
else
	python3 "${SCRIPT_DIR}/patch-ais-meson.py"
fi

echo "AIS overlay complete for ${NIXL_SRC}"
