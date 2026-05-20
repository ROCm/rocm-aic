#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Build NIXL from a checked-out tree at $NIXL_SRC (ROCm-oriented UCX + meson).
# Intended for ROCm/nixl forks; options are best-effort and can be adjusted
# per upstream meson_options.txt at your pinned ref.
#
set -euo pipefail

NIXL_SRC="${NIXL_SRC:-/tmp/nixl}"
UCX_PREFIX="${UCX_PREFIX:-/opt/rocnixl-ucx}"

if [[ ! -f "${NIXL_SRC}/meson.build" ]]; then
	echo "ERROR: ${NIXL_SRC}/meson.build not found" >&2
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
	--with-rocm=/opt/rocm \
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
MESON_EXTRA=("-Ducx_path=${UCX_PREFIX}" "-Ddisable_gds_backend=true")
if [[ -f meson_options.txt ]] && grep -q rocm_path meson_options.txt; then
	MESON_EXTRA+=("-Drocm_path=/opt/rocm")
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

python3 -c "import nixl; print('nixl import OK:', nixl.__file__)"
