#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Build NIXL from $NIXL_SRC for AMD ROCm (andyluo7/nixl amd-support + overlays).
# Meson: -Dwheel_variant=rocm -Drocm_path=/opt/rocm (see patch-rocm-meson.py).
set -euo pipefail

NIXL_SRC="${NIXL_SRC:-/tmp/nixl}"
NIXL_REQUIRE_ROCM="${NIXL_REQUIRE_ROCM:-1}"
UCX_PREFIX="${UCX_PREFIX:-/opt/rocnixl-ucx}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
AIS_PATH="${AIS_PATH:-${ROCM_PATH}}"
NIXL_ENABLE_AIS="${NIXL_ENABLE_AIS:-1}"
NIXL_INSTALL_PREFIX="${NIXL_INSTALL_PREFIX:-/opt/nixl}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "${NIXL_SRC}/meson.build" ]]; then
	echo "ERROR: ${NIXL_SRC}/meson.build not found" >&2
	exit 1
fi

if [[ "${NIXL_REQUIRE_ROCM}" == "1" ]]; then
	if [[ ! -f "${NIXL_SRC}/meson_options.txt" ]] \
		|| ! grep -q "option('wheel_variant'" "${NIXL_SRC}/meson_options.txt"; then
		echo "ERROR: ${NIXL_SRC} is not an amd-support NIXL checkout (missing wheel_variant)." >&2
		exit 1
	fi
fi

if [[ "${NIXL_ENABLE_AIS}" == "1" ]]; then
	chmod +x "${SCRIPT_DIR}/apply-ais-overlay.sh"
	NIXL_SRC="${NIXL_SRC}" "${SCRIPT_DIR}/apply-ais-overlay.sh"
fi

# Determine how this nixl checkout expects ROCm/HIP to be configured.
# Three patterns in ascending age order:
#   USE_ROCM_OPT  — ai-dynamo/nixl upstream: use_rocm string option (path or empty)
#   NATIVE_HIP    — sbates130272/nixl feat/amd-ais-mt: hip_dep auto-detected by meson
#   ROCM_PATH_OPT — andyluo7/nixl amd-support: injected rocm_path string option
_USE_ROCM_OPT=0
_NATIVE_HIP=0
_HAS_ROCM_PATH_OPT=0
if grep -q "option('use_rocm'" "${NIXL_SRC}/meson_options.txt" 2>/dev/null \
		&& ! grep -q "option('rocm_path'" "${NIXL_SRC}/meson_options.txt" 2>/dev/null; then
	_USE_ROCM_OPT=1
elif grep -q "option('rocm_path'" "${NIXL_SRC}/meson_options.txt" 2>/dev/null; then
	_HAS_ROCM_PATH_OPT=1
elif grep -q "hip_dep" "${NIXL_SRC}/meson.build" 2>/dev/null; then
	_NATIVE_HIP=1
fi
if [[ "${_USE_ROCM_OPT}" == "0" && "${_NATIVE_HIP}" == "0" && "${_HAS_ROCM_PATH_OPT}" == "0" ]]; then
	echo "ERROR: unable to determine HIP configuration for ${NIXL_SRC}: missing use_rocm/rocm_path option and hip_dep" >&2
	exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
	autoconf \
	automake \
	build-essential \
	ca-certificates \
	git \
	libaio-dev \
	libibverbs-dev \
	libltdl-dev \
	libnuma-dev \
	librdmacm-dev \
	libtool \
	liburing-dev \
	pkg-config \
	rdma-core
rm -rf /var/lib/apt/lists/*

python3 -m pip install --no-cache-dir meson ninja pybind11 tomlkit

mkdir -p /tmp/ucx-rocm
cd /tmp/ucx-rocm
if [[ ! -d ucx-src/.git ]]; then
	rm -rf ucx-src
	git clone --depth 1 https://github.com/ROCm/ucx.git -b v1.19.x ucx-src
fi
cd ucx-src
./autogen.sh
rm -rf build
mkdir build
cd build
../configure \
	--prefix="${UCX_PREFIX}" \
	--enable-shared \
	--disable-static \
	--disable-doxygen-doc \
	--enable-optimizations \
	--enable-devel-headers \
	--with-rocm="${ROCM_PATH}" \
	--with-verbs \
	--with-dm \
	--enable-mt
make -j"$(nproc)"
make install
ldconfig

export PKG_CONFIG_PATH="${UCX_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="${UCX_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${NIXL_SRC}"
rm -rf build

MESON_EXTRA=(
	"-Ducx_path=${UCX_PREFIX}"
	"-Ddisable_gds_backend=true"
	"--prefix=${NIXL_INSTALL_PREFIX}"
)

if [[ "${_USE_ROCM_OPT}" == "1" ]]; then
	# ai-dynamo/nixl upstream: use_rocm is the path (or empty for CUDA).
	MESON_EXTRA+=("-Duse_rocm=${ROCM_PATH}")
elif [[ "${_NATIVE_HIP}" == "1" ]]; then
	# sbates130272/nixl: native HIP auto-detected; wheel_variant=rocm sets Python package name.
	MESON_EXTRA+=("-Dwheel_variant=rocm")
	export ROCM_PATH="${ROCM_PATH}"
	if [[ "${NIXL_ENABLE_AIS}" == "1" && -n "${AIS_PATH:-}" ]]; then
		MESON_EXTRA+=("-Drocm_ais_path=${AIS_PATH}")
	fi
else
	# andyluo7/nixl: older injected rocm_path + wheel_variant options.
	MESON_EXTRA+=("-Dwheel_variant=rocm" "-Drocm_path=${ROCM_PATH}")
	if [[ "${NIXL_ENABLE_AIS}" == "1" && -n "${AIS_PATH:-}" ]]; then
		MESON_EXTRA+=("-Dais_path=${AIS_PATH}")
	fi
fi

meson setup build "${MESON_EXTRA[@]}"
ninja -C build
ninja -C build install
ldconfig

NIXL_PY_SITE="${NIXL_INSTALL_PREFIX}/lib/python3/dist-packages"
export PYTHONPATH="${NIXL_PY_SITE}:${PYTHONPATH:-}"

python3 -m pip uninstall -y nixl nixl-cu12 nixl-cu13 2>/dev/null || true

# Meson installs bindings as nixl_rocm. The upstream meta wheel depends on
# nixl-rocm on PyPI and its __init__.py only probes CUDA backends, so pip
# install of the meta wheel or repo root pyproject.toml is wrong on ROCm.
if [[ ! -d "${NIXL_PY_SITE}/nixl_rocm" ]]; then
	echo "ERROR: ${NIXL_PY_SITE}/nixl_rocm not found after meson install" >&2
	exit 1
fi

mkdir -p "${NIXL_PY_SITE}/nixl"
cat > "${NIXL_PY_SITE}/nixl/__init__.py" <<'PY'
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""ROCm shim: meson installs nixl_rocm; consumers import nixl."""

import importlib
import sys

_pkg = importlib.import_module("nixl_rocm")

for sub_name in ("_api", "_bindings", "_utils", "logging"):
    module = importlib.import_module(f"{_pkg.__name__}.{sub_name}")
    sys.modules[f"nixl.{sub_name}"] = module
    setattr(sys.modules[__name__], sub_name, module)
    for attr in dir(module):
        if not attr.startswith("_"):
            setattr(sys.modules[__name__], attr, getattr(module, attr))
PY

export NIXL_PLUGIN_DIR="${NIXL_INSTALL_PREFIX}/lib/x86_64-linux-gnu/plugins"
if [[ ! -d "${NIXL_PLUGIN_DIR}" ]]; then
	NIXL_PLUGIN_DIR="${NIXL_INSTALL_PREFIX}/lib/nixl/plugins"
fi
export LD_LIBRARY_PATH="${NIXL_INSTALL_PREFIX}/lib/x86_64-linux-gnu:${NIXL_INSTALL_PREFIX}/lib:${UCX_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

python3 -c "import nixl; print('nixl import OK:', nixl.__file__)"

if [[ "${NIXL_ENABLE_AIS}" == "1" ]]; then
	for plug in AIS AIS_MT; do
		found="$(find "${NIXL_SRC}/build" -name "libplugin_${plug}.so" 2>/dev/null | head -1 || true)"
		if [[ -n "${found}" ]]; then
			echo "PASS: ${found}"
		else
			echo "WARN: libplugin_${plug}.so not found (hipFile may be missing)"
		fi
	done
fi

echo "NIXL build complete prefix=${NIXL_INSTALL_PREFIX}"
