#!/usr/bin/env bash
#
# Mount a WEKA driver squashfs, copy to a writable dir, and build the 
# wekafsio.ko and wekafsgw.ko kernel modules. Requires root for mount. See
# README for details.
#
set -e
set -u

readonly DEFAULT_SQUASHFS="weka-drops/weka-driver-995ca4aed98e13aa.squashfs"
SCRIPT_ROOT="$(cd "$(dirname "$0")" && pwd)"
WEKA_MOUNT_DIR_CREATED=0
WEKA_BUILD_WORK_DIR_CREATED=0
EXIT_CODE=0

die() {
  echo "$*" >&2
  exit 1
}

cleanup() {
  local code=$?
  if [[ -n "${WEKA_LEAVE_MOUNTED:-}" ]]; then
    echo "WEKA_LEAVE_MOUNTED set; leaving mount at ${WEKA_MOUNT_DIR:-}" >&2
  elif [[ -n "${WEKA_MOUNT_DIR:-}" ]] && mountpoint -q "${WEKA_MOUNT_DIR}" 2>/dev/null; then
    umount "${WEKA_MOUNT_DIR}" || true
    if [[ "$WEKA_MOUNT_DIR_CREATED" -eq 1 ]] && [[ -d "${WEKA_MOUNT_DIR}" ]]; then
      rmdir "${WEKA_MOUNT_DIR}" 2>/dev/null || true
    fi
  fi
  exit "${EXIT_CODE:-$code}"
}

# --- Resolve paths -----------------------------------------------------------
if [[ -n "${WEKA_SQUASHFS:-}" ]]; then
  if [[ "${WEKA_SQUASHFS}" != /* ]]; then
    SQUASHFS_PATH="${SCRIPT_ROOT}/${WEKA_SQUASHFS}"
  else
    SQUASHFS_PATH="${WEKA_SQUASHFS}"
  fi
else
  SQUASHFS_PATH="${SCRIPT_ROOT}/${DEFAULT_SQUASHFS}"
fi

if [[ ! -f "${SQUASHFS_PATH}" ]] || [[ ! -r "${SQUASHFS_PATH}" ]]; then
  die "Squashfs file not found or not readable: ${SQUASHFS_PATH}"
fi

# --- Root check --------------------------------------------------------------
if [[ "$(id -u)" -ne 0 ]]; then
  die "This script must be run as root (mount requires it)."
fi

# --- Mount point ------------------------------------------------------------
if [[ -n "${WEKA_MOUNT_DIR:-}" ]]; then
  if [[ ! -d "${WEKA_MOUNT_DIR}" ]]; then
    mkdir -p "${WEKA_MOUNT_DIR}" || die "Cannot create WEKA_MOUNT_DIR: ${WEKA_MOUNT_DIR}"
  fi
  WEKA_MOUNT_DIR_CREATED=0
else
  WEKA_MOUNT_DIR="$(mktemp -d /tmp/wekafs-mount.XXXXXX)"
  WEKA_MOUNT_DIR_CREATED=1
fi

trap cleanup EXIT

# --- Mount -------------------------------------------------------------------
mount -t squashfs "${SQUASHFS_PATH}" "${WEKA_MOUNT_DIR}" \
  || die "Failed to mount ${SQUASHFS_PATH} on ${WEKA_MOUNT_DIR}"

# --- Work dir for copy + build -----------------------------------------------
if [[ -n "${WEKA_BUILD_WORK_DIR:-}" ]]; then
  if [[ ! -d "${WEKA_BUILD_WORK_DIR}" ]]; then
    mkdir -p "${WEKA_BUILD_WORK_DIR}" || die "Cannot create WEKA_BUILD_WORK_DIR"
  fi
  WEKA_BUILD_WORK_DIR_CREATED=0
else
  WEKA_BUILD_WORK_DIR="$(mktemp -d /tmp/wekafs-build.XXXXXX)"
  WEKA_BUILD_WORK_DIR_CREATED=1
fi

# --- Copy --------------------------------------------------------------------
cp -a "${WEKA_MOUNT_DIR}/"* "${WEKA_BUILD_WORK_DIR}/" \
  || die "Failed to copy from ${WEKA_MOUNT_DIR} to ${WEKA_BUILD_WORK_DIR}"

# --- Kernel build dir -------------------------------------------------------
WEKA_KERNEL_BUILD_DIR="${WEKA_KERNEL_BUILD_DIR:-/lib/modules/$(uname -r)/build}"
if [[ ! -d "${WEKA_KERNEL_BUILD_DIR}" ]]; then
  echo "Kernel build dir missing: ${WEKA_KERNEL_BUILD_DIR}" >&2
  echo "Install kernel headers for your target kernel (e.g. linux-headers-*)" >&2
  if [[ -d /lib/modules ]]; then
    echo "Available under /lib/modules: $(ls /lib/modules 2>/dev/null | tr '\n' ' ')" >&2
  fi
  die "Set WEKA_KERNEL_BUILD_DIR to an existing path, or install headers."
fi

# --- Build -------------------------------------------------------------------
if ! (cd "${WEKA_BUILD_WORK_DIR}" && ./build.sh "${WEKA_KERNEL_BUILD_DIR}" modules); then
  EXIT_CODE=1
  die "Build failed in ${WEKA_BUILD_WORK_DIR}"
fi

# Change ownership of the build dir to the caller (when run via sudo).
if [[ -n "${SUDO_UID:-}" ]] && [[ -n "${SUDO_GID:-}" ]]; then
  chown -R "${SUDO_UID}:${SUDO_GID}" "${WEKA_BUILD_WORK_DIR}"
fi

echo "Build complete. Modules in: ${WEKA_BUILD_WORK_DIR}"
EXIT_CODE=0
