#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Fail-closed gates for the canonical LMCache wheel (the HIP c_ops extension
# and the host-only Mooncake backend in ONE wheel) and for the source-built
# Mooncake ROCm package.  Every gate exits non-zero on failure, so a failing
# gate stops the image build.
#
# Run twice by the build: `wheels` before anything is installed (archive
# membership cannot be faked by an install), `runtime` after the install.
# The script is kept in the image so the same gates can be re-run against a
# built image without rebuilding:
#
#   docker run --rm --entrypoint bash <image> \
#       /usr/local/bin/aic-mooncake-gates.sh runtime
#
# Usage:
#   aic-mooncake-gates.sh wheels <lmcache-wheel> <mooncake-wheel>
#   aic-mooncake-gates.sh runtime
set -euo pipefail

MOONCAKE_PREFIX="${MOONCAKE_PREFIX:-/opt/mooncake-sdk}"
# shellcheck disable=SC2016 # $ORIGIN must remain literal in the ELF RPATH.
WHEEL_MOONCAKE_RPATH='$ORIGIN/../mooncake:$ORIGIN/../mooncake_transfer_engine_rocm.libs'

die() {
	echo "ERROR: $*" >&2
	exit 1
}

# --- wheel archive membership (must run BEFORE the wheel is installed) -------
# Validate both archives before either can be masked by an installed package.
gate_wheels() {
	local lmcache_wheel="${1:-}"
	local mooncake_wheel="${2:-}"
	[[ -f "${lmcache_wheel}" ]] || die "LMCache wheel not found: ${lmcache_wheel}"
	[[ -f "${mooncake_wheel}" ]] || die "Mooncake wheel not found: ${mooncake_wheel}"
	python3 - "${lmcache_wheel}" "${mooncake_wheel}" <<'PY'
import sys
import zipfile

lmcache_wheel, mooncake_wheel = sys.argv[1:]

def members(path):
    with zipfile.ZipFile(path) as archive:
        return archive.namelist()

lmcache = members(lmcache_wheel)
for prefix in ("lmcache/c_ops", "lmcache/lmcache_mooncake"):
    if not any(name.startswith(prefix) and name.endswith(".so") for name in lmcache):
        raise SystemExit("ERROR: %s is missing %s*.so" % (lmcache_wheel, prefix))

mooncake = members(mooncake_wheel)
mooncake_runtime = (
    "mooncake/engine.so",
    "mooncake/store.so",
    "mooncake/mooncake_master",
    "mooncake/libmooncake_store.so",
    "mooncake/libmooncake_common.so",
    "mooncake/libtransfer_engine.so",
    "mooncake/libasio.so",
    "mooncake/libetcd_wrapper.so",
)
for name in mooncake_runtime:
    if name not in mooncake:
        raise SystemExit("ERROR: %s is missing %s" % (mooncake_wheel, name))

print("  LMCache native members:")
for name in sorted(name for name in lmcache if name.endswith(".so")):
    print("    %s" % name)
print("  Mooncake runtime members:")
for name in mooncake_runtime:
    print("    %s" % name)
PY
	echo "PASS: wheel archives contain both LMCache extensions and Mooncake runtime"
}

# --- published wheels load without the image-only SDK -----------------------
# The LMCache extension is linked against the SDK while compiling, but the
# exported wheel must resolve the same library from the companion Mooncake ROCm
# wheel. Install both archives into an isolated target, hide the SDK, and prove
# the module path, RPATH and loader result all point at that target.
gate_wheel_install() (
	local lmcache_wheel="${1:-}"
	local mooncake_wheel="${2:-}"
	local target hidden extension rpath output resolved expected master
	local torch_lib external_ld_library_path
	target="$(mktemp -d)"
	hidden="${MOONCAKE_PREFIX}.wheel-gate-hidden.$$"
	[[ -d "${MOONCAKE_PREFIX}" ]] || die "Mooncake SDK not found: ${MOONCAKE_PREFIX}"
	[[ ! -e "${hidden}" ]] || die "temporary SDK path already exists: ${hidden}"
	# shellcheck disable=SC2317,SC2329 # Invoked by the EXIT trap below.
	cleanup() {
		if [[ -e "${hidden}" && ! -e "${MOONCAKE_PREFIX}" ]]; then
			mv "${hidden}" "${MOONCAKE_PREFIX}"
		fi
		rm -rf "${target}"
	}
	trap cleanup EXIT

	python3 -m pip install --no-cache-dir --no-deps --target "${target}" \
		"${mooncake_wheel}" "${lmcache_wheel}"
	mv "${MOONCAKE_PREFIX}" "${hidden}"
	# Torch and the ROCm runtime are documented host prerequisites, not contents
	# of either wheel. Expose only those external loader roots; Mooncake itself
	# must still resolve through the wheel-relative RPATH checked below.
	torch_lib="$(python3 -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")')"
	[[ -d "${torch_lib}" ]] || die "installed Torch library directory is missing: ${torch_lib}"
	[[ -e /opt/rocm/lib/libamdhip64.so.7 ]] || die "ROCm runtime library is missing: libamdhip64.so.7"
	external_ld_library_path="${torch_lib}:/opt/rocm/lib"

	extension="$(find "${target}/lmcache" -maxdepth 1 -type f \
		-name 'lmcache_mooncake*.so' -print -quit)"
	[[ -n "${extension}" ]] || die "clean install is missing lmcache_mooncake*.so"
	rpath="$(patchelf --print-rpath "${extension}")"
	[[ "${rpath}" == "${WHEEL_MOONCAKE_RPATH}" ]] || \
		die "LMCache Mooncake RPATH is ${rpath}, expected ${WHEEL_MOONCAKE_RPATH}"

	output="$(LD_LIBRARY_PATH="${external_ld_library_path}" ldd "${extension}" 2>&1)" || {
		echo "${output}" >&2
		die "ldd failed for clean-installed LMCache Mooncake extension"
	}
	if grep -q "not found" <<<"${output}"; then
		echo "${output}" >&2
		die "clean-installed LMCache Mooncake extension has unresolved libraries"
	fi
	resolved="$(awk '$1 ~ /^libmooncake_store\.so/ && $2 == "=>" { print $3; exit }' \
		<<<"${output}")"
	[[ -n "${resolved}" && -e "${resolved}" ]] || {
		echo "${output}" >&2
		die "LMCache extension did not resolve an existing libmooncake_store"
	}
	expected="${target}/mooncake/libmooncake_store.so"
	resolved="$(realpath "${resolved}")"
	expected="$(realpath "${expected}")"
	[[ "${resolved}" == "${expected}" ]] || {
		echo "${output}" >&2
		die "LMCache extension resolved libmooncake_store outside the companion wheel"
	}

	LD_LIBRARY_PATH="${external_ld_library_path}" \
		PYTHONPATH="${target}" PYTHONNOUSERSITE=1 \
		python3 - "${target}" <<'PY'
import importlib
import importlib.util
import pathlib
import sys

import torch  # noqa: F401 — must be imported before lmcache.c_ops

root = pathlib.Path(sys.argv[1]).resolve()

# LMCache installs a compatibility shim at lmcache.c_ops on CPU-only hosts.
# Load the wheel's extension file directly so this GPU-less build gate still
# proves that the native artifact itself can be initialized.
native_path = next((root / "lmcache").glob("c_ops*.so"), None)
if native_path is None:
    raise SystemExit("ERROR: clean install is missing c_ops*.so")
spec = importlib.util.spec_from_file_location("lmcache.c_ops", native_path)
if spec is None or spec.loader is None:
    raise SystemExit("ERROR: cannot create an import spec for %s" % native_path)
native = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = native
spec.loader.exec_module(native)
loaded_native_path = pathlib.Path(native.__file__).resolve()
if not loaded_native_path.is_relative_to(root):
    raise SystemExit(
        "ERROR: lmcache.c_ops loaded from %s, outside %s"
        % (loaded_native_path, root)
    )
print("  clean import: lmcache.c_ops -> %s" % loaded_native_path)

for name in (
    "lmcache.lmcache_mooncake",
    "mooncake.engine",
    "mooncake.store",
):
    module = importlib.import_module(name)
    path = pathlib.Path(module.__file__).resolve()
    if not path.is_relative_to(root):
        raise SystemExit("ERROR: %s loaded from %s, outside %s" % (name, path, root))
    print("  clean import: %s -> %s" % (name, path))
PY
	master="${target}/mooncake/mooncake_master"
	[[ -x "${master}" ]] || die "clean install is missing executable mooncake_master"
	LD_LIBRARY_PATH="${external_ld_library_path}" "${master}" --version
	echo "PASS: published LMCache and Mooncake wheels load without the image SDK"
)

# --- imports ----------------------------------------------------------------
gate_imports() {
	python3 <<'PY'
import importlib
from importlib.metadata import version

if version("lmcache") != "0.5.3":
    raise SystemExit("ERROR: expected lmcache 0.5.3, got %s" % version("lmcache"))

for name in (
    "lmcache.c_ops",
    "lmcache.lmcache_mooncake",
    "mooncake.engine",
    "mooncake.store",
):
    importlib.import_module(name)
    print("  import: %s" % name)
PY
	echo "PASS: c_ops, lmcache_mooncake, mooncake.engine and mooncake.store import"
}

# --- the MP adapters needed by the connector come from LMCache ---------------
# lmcache_mp_connector silently falls back to the copy vendored inside the
# serving engine if its LMCache adapter imports fail. Import the same module and
# record types used by that try block so a wheel that lost them fails here. Do
# not import the connector itself in this GPU-less build gate: vLLM's ROCm
# platform initialization requires a visible GPU. The hardware smoke imports
# the connector and checks its selected classes in the real runtime.
gate_adapter_identity() {
	python3 <<'PY'
import importlib

module = importlib.import_module(
    "lmcache.integration.vllm.vllm_multi_process_adapter"
)
expected = "lmcache.integration.vllm.vllm_multi_process_adapter"
for attr in (
    "LMCacheMPSchedulerAdapter",
    "LMCacheMPWorkerAdapter",
    "LoadStoreOp",
    "ParallelStrategy",
    "send_lmcache_request",
):
    if not hasattr(module, attr):
        raise SystemExit("ERROR: %s missing from %s" % (attr, expected))

for attr in ("LMCacheMPSchedulerAdapter", "LMCacheMPWorkerAdapter"):
    cls = getattr(module, attr, None)
    if cls.__module__ != expected:
        raise SystemExit(
            "ERROR: %s resolved to %s, expected %s"
            % (attr, cls.__module__, expected)
        )
    print("  adapter: %s -> %s" % (attr, cls.__module__))

custom_types = importlib.import_module("lmcache.v1.multiprocess.custom_types")
try:
    allocation_record = custom_types.RequestAllocationRecord
except AttributeError:
    try:
        allocation_record = custom_types.BlockAllocationRecord
    except AttributeError as error:
        raise SystemExit(
            "ERROR: connector-compatible allocation record missing from %s"
            % custom_types.__name__
        ) from error
print("  allocation record: %s" % allocation_record.__name__)
PY
	echo "PASS: connector MP adapter imports resolve to lmcache modules"
}

# --- the packaged master binary runs ----------------------------------------
gate_master() {
	local master
	master="$(python3 -c 'import mooncake, pathlib; print(pathlib.Path(mooncake.__path__[0]) / "mooncake_master")')"
	[[ -x "${master}" ]] || die "packaged mooncake_master not found at ${master}"
	"${master}" --version
	echo "PASS: packaged mooncake_master reports its version"
}

# --- no unresolved shared libraries -----------------------------------------
gate_ldd() {
	local target
	local status=0
	local output resolved expected
	local targets torch_lib loader_path
	torch_lib="$(python3 -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")')"
	[[ -d "${torch_lib}" ]] || die "installed Torch library directory is missing: ${torch_lib}"
	loader_path="${torch_lib}"
	if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
		loader_path="${loader_path}:${LD_LIBRARY_PATH}"
	fi
	targets="$(mktemp)"
	if ! ldd_targets >"${targets}"; then
		rm -f "${targets}"
		die "could not resolve the shared-object target list"
	fi
	while IFS= read -r target; do
		[[ -n "${target}" ]] || continue
		[[ -e "${target}" ]] || die "expected shared object is missing: ${target}"
		if ! output="$(LD_LIBRARY_PATH="${loader_path}" ldd "${target}" 2>&1)"; then
			echo "ldd failed for ${target}:" >&2
			echo "${output}" >&2
			status=1
		elif grep -q "not found" <<<"${output}"; then
			echo "unresolved libraries in ${target}:" >&2
			grep "not found" <<<"${output}" >&2
			status=1
		else
			if [[ "$(basename "${target}")" == lmcache_mooncake*.so ]]; then
				resolved="$(awk '$1 ~ /^libmooncake_store\.so/ && $2 == "=>" { print $3; exit }' \
					<<<"${output}")"
				expected="$(python3 -c 'import mooncake, pathlib; print(pathlib.Path(mooncake.__path__[0]) / "libmooncake_store.so")')"
				if [[ -z "${resolved}" || ! -e "${resolved}" || \
					"$(realpath "${resolved}")" != "$(realpath "${expected}")" ]]; then
					echo "${output}" >&2
					die "installed LMCache extension did not resolve the companion-wheel libmooncake_store"
				fi
				echo "  companion wheel: $(realpath "${resolved}")"
			fi
			echo "  resolved: ${target}"
		fi
	done <"${targets}"
	rm -f "${targets}"
	[[ "${status}" -eq 0 ]] || die "unresolved shared libraries, see above"
	echo "PASS: every checked shared object resolves its libraries"
}

ldd_targets() {
	python3 <<'PY' || return 1
import pathlib

import lmcache
import mooncake

for package, patterns in (
    (lmcache, ("c_ops*.so", "lmcache_mooncake*.so")),
    (mooncake, ("engine*.so", "store*.so")),
):
    root = pathlib.Path(package.__path__[0])
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if not matches:
            raise SystemExit("ERROR: no %s under %s" % (pattern, root))
        for match in matches:
            print(match)
PY
	local master
	master="$(python3 -c 'import mooncake, pathlib; print(pathlib.Path(mooncake.__path__[0]) / "mooncake_master")')" || return 1
	printf '%s\n' \
		"${MOONCAKE_PREFIX}/lib/libmooncake_store.so" \
		"${MOONCAKE_PREFIX}/lib/libtransfer_engine.so" \
		"${MOONCAKE_PREFIX}/lib/libmooncake_common.so" \
		"${MOONCAKE_PREFIX}/lib/libasio.so" \
		"${MOONCAKE_PREFIX}/lib/libetcd_wrapper.so" \
		"${master}"
}

# --- Mooncake's declared Python dependencies are satisfied -------------------
# AIC intentionally removes CUDA-only packages such as cufile-python on ROCm,
# although LMCache declares them unconditionally. A whole-environment
# `pip check` therefore cannot be a valid gate here. Keep its useful signal,
# but fail only on the distribution introduced by this change; the native
# LMCache and Mooncake imports above cover the two extensions themselves.
gate_pip_check() {
	local report
	report="$(python3 -m pip check 2>&1 || true)"
	if grep -Eqi '^mooncake[-_]transfer[-_]engine[-_]rocm ' <<<"${report}"; then
		echo "Mooncake dependency check failed:" >&2
		grep -Ei '^mooncake[-_]transfer[-_]engine[-_]rocm ' <<<"${report}" >&2
		die "Mooncake package metadata is inconsistent"
	fi
	python3 -c 'import msgpack; from importlib.metadata import version; print("  dependency: msgpack " + version("msgpack"))'
	echo "PASS: Mooncake's declared Python dependencies are installed"
}

gate_runtime() {
	gate_imports
	gate_adapter_identity
	gate_master
	gate_ldd
	gate_pip_check
	echo "PASS: runtime gates complete"
}

main() {
	local command="${1:-}"
	shift || true
	case "${command}" in
	wheels)
		gate_wheels "$@"
		gate_wheel_install "$@"
		;;
	runtime) gate_runtime ;;
	*) die "usage: $0 {wheels <lmcache-wheel> <mooncake-wheel>|runtime}" ;;
	esac
}

main "$@"
