#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Build NIXL from $NIXL_SRC for AMD ROCm.  The checkout arrives pre-patched with
# nixl-rocm-ais-mt.patch (native HIP + the AIS_MT plugin), applied in the
# Dockerfile.  Meson: -Dwheel_variant=rocm -Drocm_ais_path=$AIS_PATH.
set -euo pipefail

NIXL_SRC="${NIXL_SRC:-/tmp/nixl}"
NIXL_REQUIRE_ROCM="${NIXL_REQUIRE_ROCM:-1}"
UCX_PREFIX="${UCX_PREFIX:-/opt/rocnixl-ucx}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
AIS_PATH="${AIS_PATH:-${ROCM_PATH}}"
NIXL_INSTALL_PREFIX="${NIXL_INSTALL_PREFIX:-/opt/nixl}"

BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

if [[ ! -f "${NIXL_SRC}/meson.build" ]]; then
	echo "ERROR: ${NIXL_SRC}/meson.build not found" >&2
	exit 1
fi

if [[ "${NIXL_REQUIRE_ROCM}" == "1" ]]; then
	if [[ ! -f "${NIXL_SRC}/meson_options.txt" ]] \
		|| ! grep -q "option('wheel_variant'" "${NIXL_SRC}/meson_options.txt"; then
		echo "ERROR: ${NIXL_SRC} is missing the wheel_variant meson option — wrong ref?" >&2
		exit 1
	fi
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
	autoconf \
	automake \
	build-essential \
	ca-certificates \
	cmake \
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
make -j"${BUILD_JOBS}"
make install
ldconfig

export PKG_CONFIG_PATH="${UCX_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="${UCX_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${NIXL_SRC}"
rm -rf build

MESON_EXTRA=(
	"-Dwheel_variant=rocm"
	"-Drocm_ais_path=${AIS_PATH}"
	"-Ducx_path=${UCX_PREFIX}"
	"-Ddisable_gds_backend=true"
	"--prefix=${NIXL_INSTALL_PREFIX}"
)

meson setup build "${MESON_EXTRA[@]}"
ninja -C build -j"${BUILD_JOBS}"
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

# The AIS_MT plugin (LMCache's l2-adapter backend) is mandatory; its absence
# means hipFile wasn't found at NIXL build time (AIS_MT links hipFileRead/Write).
found="$(find "${NIXL_SRC}/build" -name "libplugin_AIS_MT.so" 2>/dev/null | head -1 || true)"
if [[ -n "${found}" ]]; then
	echo "PASS: ${found}"
else
	echo "ERROR: libplugin_AIS_MT.so not built (hipFile missing at NIXL build time?)" >&2
	exit 1
fi

# The NIXL Prometheus telemetry exporter is MANDATORY here: builds without a
# working plugin are rejected (like the AIS_MT check above).  NIXL meson builds
# it only when the prometheus-cpp CMake subproject resolves (see
# src/plugins/telemetry/prometheus/meson.build; wrap = jupp0r/prometheus-cpp
# v1.3.0, ENABLE_PULL=ON/PUSH=OFF, USE_THIRDPARTY_LIBRARIES=ON -> bundled
# civetweb).  Enable it at runtime on the process that runs the NIXL agent (the
# LMCache server in this stack) with:
#     NIXL_TELEMETRY_ENABLE=y NIXL_TELEMETRY_EXPORTER=prometheus \
#     NIXL_TELEMETRY_PROMETHEUS_PORT=19090     # serves /metrics on :19090
prom_plugin="$(find "${NIXL_SRC}/build" -name "libtelemetry_exporter_prometheus.so" 2>/dev/null | head -1 || true)"
if [[ -z "${prom_plugin}" ]]; then
	echo "ERROR: prometheus telemetry exporter plugin NOT built (prometheus-cpp subproject unavailable at NIXL build time); this build is not permitted without it" >&2
	exit 1
fi
echo "PASS: prometheus telemetry exporter plugin: ${prom_plugin}"
# meson installs the plugin itself, but the prometheus-cpp/civetweb shared libs
# from the CMake subproject build tree are NOT installed by `ninja install`.
# Copy them onto a dir already on the runtime LD_LIBRARY_PATH so the plugin can
# dlopen its deps at load time.
install -Dm0755 "${prom_plugin}" \
	"${NIXL_PLUGIN_DIR}/$(basename "${prom_plugin}")"
while IFS= read -r _lib; do
	[[ -n "${_lib}" ]] || continue
	install -Dm0755 "${_lib}" \
		"${NIXL_INSTALL_PREFIX}/lib/x86_64-linux-gnu/$(basename "${_lib}")"
done < <(find "${NIXL_SRC}/build" \
	\( -name 'libprometheus-cpp*.so*' -o -name 'libcivetweb*.so*' \) 2>/dev/null)
ldconfig
# A plugin that can't resolve its shared libs is as good as absent -> fatal.
if command -v ldd >/dev/null 2>&1; then
	if ldd "${NIXL_PLUGIN_DIR}/$(basename "${prom_plugin}")" 2>/dev/null \
		| grep -q "not found"; then
		echo "ERROR: prometheus telemetry plugin has unresolved shared libs:" >&2
		ldd "${NIXL_PLUGIN_DIR}/$(basename "${prom_plugin}")" \
			| grep "not found" >&2 || true
		exit 1
	fi
fi

echo "NIXL build complete prefix=${NIXL_INSTALL_PREFIX}"

# ----- Optional: build a pip-installable ROCm wheel (nixl-rocm) --------------
# When NIXL_BUILD_WHEEL=1, also emit a wheel into NIXL_WHEEL_DIR (default
# /wheels) via meson-python, reusing the SAME meson args resolved above
# (MESON_EXTRA, minus --prefix which meson-python manages itself).  The wheel is
# named nixl-rocm (upstream pyproject defaults to nixl-cu12) and bundles libnixl
# + the NIXL plugins/UCX libs; the ROCm runtime (libamdhip64, hipFile) stays an
# external dependency of the target environment -- see the AIC README.
#
# CXX must NOT be hipcc for this build: meson would try to use it as the host C++
# compiler and hipcc fails meson's compiler sanity check.  The Dockerfile keeps
# CXX=hipcc scoped to the LMCache layer, but `env -u CXX` here makes the wheel
# build robust regardless of the caller's environment.
if [[ "${NIXL_BUILD_WHEEL:-0}" == "1" ]]; then
	NIXL_WHEEL_DIR="${NIXL_WHEEL_DIR:-/wheels}"
	mkdir -p "${NIXL_WHEEL_DIR}"
	python3 -m pip install --no-cache-dir meson-python patchelf
	cd "${NIXL_SRC}"
	if [[ -x contrib/tomlutil.py ]]; then
		./contrib/tomlutil.py --wheel-name nixl-rocm pyproject.toml
	fi
	wheel_setup_args=()
	for _arg in "${MESON_EXTRA[@]}"; do
		[[ "${_arg}" == --prefix=* ]] && continue
		wheel_setup_args+=("-Csetup-args=${_arg}")
	done
	wheel_setup_args+=("-Ccompile-args=-j${BUILD_JOBS}")
	env -u CXX ROCM_PATH="${ROCM_PATH}" \
		uv build --wheel --no-build-isolation \
			--out-dir "${NIXL_WHEEL_DIR}" "${wheel_setup_args[@]}"
	_nixl_whl="$(find "${NIXL_WHEEL_DIR}" -maxdepth 1 -name 'nixl*rocm*.whl' | head -1)"
	if [[ -z "${_nixl_whl}" ]]; then
		echo "ERROR: NIXL_BUILD_WHEEL=1 but no nixl*rocm*.whl in ${NIXL_WHEEL_DIR}" >&2
		exit 1
	fi
	echo "PASS: NIXL wheel built -> ${_nixl_whl}"
fi
