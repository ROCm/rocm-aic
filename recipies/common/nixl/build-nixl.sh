#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Build NIXL from $NIXL_SRC for AMD ROCm (andyluo7/nixl amd-support + AIS overlay).
# Meson: -Duse_rocm=/opt/rocm (path string on amd-support branch).
set -euo pipefail

NIXL_SRC="${NIXL_SRC:-/tmp/nixl}"
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

if [[ "${NIXL_ENABLE_AIS}" == "1" ]]; then
	chmod +x "${SCRIPT_DIR}/apply-ais-overlay.sh"
	NIXL_SRC="${NIXL_SRC}" "${SCRIPT_DIR}/apply-ais-overlay.sh"
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
	"-Duse_rocm=${ROCM_PATH}"
	"-Ducx_path=${UCX_PREFIX}"
	"-Ddisable_gds_backend=true"
	"--prefix=${NIXL_INSTALL_PREFIX}"
)

if [[ "${NIXL_ENABLE_AIS}" == "1" && -n "${AIS_PATH}" ]]; then
	MESON_EXTRA+=("-Dais_path=${AIS_PATH}")
fi

meson setup build "${MESON_EXTRA[@]}"
ninja -C build
ninja -C build install
ldconfig

shopt -s nullglob || true
for wheel in "${NIXL_SRC}"/build/src/bindings/python/nixl-meta/nixl-*-py3-none-any.whl; do
	[[ -f "${wheel}" ]] || continue
	python3 -m pip install --no-cache-dir "${wheel}" && break
done
shopt -u nullglob || true

if [[ -f "${NIXL_SRC}/pyproject.toml" ]] && ! python3 -c "import nixl" 2>/dev/null; then
	python3 -m pip install --no-cache-dir "${NIXL_SRC}" || true
fi

export NIXL_PLUGIN_DIR="${NIXL_INSTALL_PREFIX}/lib/x86_64-linux-gnu/nixl/plugins"
if [[ ! -d "${NIXL_PLUGIN_DIR}" ]]; then
	NIXL_PLUGIN_DIR="${NIXL_INSTALL_PREFIX}/lib/nixl/plugins"
fi
export LD_LIBRARY_PATH="${NIXL_INSTALL_PREFIX}/lib:${UCX_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

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
