#!/usr/bin/env bash
#
# Verify that amdgpu-dkms-tool.py correctly propagates the
# new version to every location inside the repacked .deb.
#
# Usage:
#   bash test-repack.sh <input.deb>
#
# The test:
#   1. Repacks the .deb with --suffix .testver
#   2. Extracts the result and checks all five version locations
#   3. Uses fakeroot to install the .deb into a temp root and
#      verifies the source tree lands at the correct path
#   4. Cleans up all temp files
#
# Requires: python3, dpkg-deb, fakeroot, grep
#
set -euo pipefail

SUFFIX=".testver"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOL="${SCRIPT_DIR}/../amdgpu-dkms-tool.py"
PASS=0
FAIL=0

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input.deb>" >&2
    exit 1
fi

INPUT_DEB="$(realpath "$1")"
if [ ! -f "$INPUT_DEB" ]; then
    echo "Error: file not found: $INPUT_DEB" >&2
    exit 1
fi

# shellcheck disable=SC2317
cleanup() {
    rm -rf "${WORK_DIR:-}" "${VERIFY_DIR:-}" "${INSTALL_DIR:-}" "${ORIG_DIR:-}"
    rm -f "${OUT_DEB:-}"
}
trap cleanup EXIT

WORK_DIR="$(mktemp -d -t test-repack-work.XXXXXX)"
VERIFY_DIR="$(mktemp -d -t test-repack-verify.XXXXXX)"
INSTALL_DIR="$(mktemp -d -t test-repack-install.XXXXXX)"
OUT_DEB="${WORK_DIR}/repacked.deb"

pass() { PASS=$((PASS + 1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1" >&2; }

# --- Discover the original DKMS version ---

ORIG_DIR="$(mktemp -d -t test-repack-orig.XXXXXX)"
dpkg-deb -R "$INPUT_DEB" "$ORIG_DIR" >/dev/null 2>&1

mapfile -t dkms_dirs < <(
    find "$ORIG_DIR/usr/src" -maxdepth 1 \
        -type d -name 'amdgpu-*'
)
if [ "${#dkms_dirs[@]}" -eq 0 ]; then
    echo "Error: no amdgpu-* DKMS directory found under $ORIG_DIR/usr/src" >&2
    exit 1
fi
if [ "${#dkms_dirs[@]}" -gt 1 ]; then
    echo "Error: multiple amdgpu-* DKMS directories found under $ORIG_DIR/usr/src: ${dkms_dirs[*]}" >&2
    exit 1
fi
OLD_DKMS_DIR="${dkms_dirs[0]}"
OLD_DKMS_VER="${OLD_DKMS_DIR##*/amdgpu-}"

OLD_CONTROL_VER=$(
    grep -m1 '^Version:' "$ORIG_DIR/DEBIAN/control" \
        | sed 's/^Version:[[:space:]]*//'
)
NEW_CONTROL_VER="${OLD_CONTROL_VER}${SUFFIX}"
rm -rf "$ORIG_DIR"

echo "=== Test parameters ==="
echo "  Input:            $INPUT_DEB"
echo "  Old control ver:  $OLD_CONTROL_VER"
echo "  Old DKMS ver:     $OLD_DKMS_VER"
echo "  New version:      $NEW_CONTROL_VER"
echo ""

# --- Step 1: Run the repack tool ---

echo "=== Running amdgpu-dkms-tool.py ==="
python3 "$TOOL" "$INPUT_DEB" \
    --suffix "$SUFFIX" \
    --output "$OUT_DEB"
echo ""

if [ ! -f "$OUT_DEB" ]; then
    echo "FATAL: repack tool did not produce output" >&2
    exit 1
fi

# --- Step 2: Extract and verify version strings ---

echo "=== Verifying version strings ==="
dpkg-deb -R "$OUT_DEB" "$VERIFY_DIR" >/dev/null 2>&1

# 2a. DEBIAN/control
CTRL_VER=$(
    grep -m1 '^Version:' "$VERIFY_DIR/DEBIAN/control" \
        | sed 's/^Version:[[:space:]]*//'
)
if [ "$CTRL_VER" = "$NEW_CONTROL_VER" ]; then
    pass "DEBIAN/control Version = $CTRL_VER"
else
    fail "DEBIAN/control Version = $CTRL_VER (expected $NEW_CONTROL_VER)"
fi

# 2b. DEBIAN/postinst CVERSION
if [ -f "$VERIFY_DIR/DEBIAN/postinst" ]; then
    POST_VER=$(
        grep -m1 '^CVERSION=' \
            "$VERIFY_DIR/DEBIAN/postinst" \
            | sed 's/^CVERSION=//'
    )
    if [ "$POST_VER" = "$NEW_CONTROL_VER" ]; then
        pass "DEBIAN/postinst CVERSION = $POST_VER"
    else
        fail "DEBIAN/postinst CVERSION = $POST_VER (expected $NEW_CONTROL_VER)"
    fi
else
    fail "DEBIAN/postinst not found"
fi

# 2c. DEBIAN/prerm VERSION
if [ -f "$VERIFY_DIR/DEBIAN/prerm" ]; then
    PRERM_VER=$(
        grep -m1 '^VERSION=' \
            "$VERIFY_DIR/DEBIAN/prerm" \
            | sed 's/^VERSION=//'
    )
    if [ "$PRERM_VER" = "$NEW_CONTROL_VER" ]; then
        pass "DEBIAN/prerm VERSION = $PRERM_VER"
    else
        fail "DEBIAN/prerm VERSION = $PRERM_VER (expected $NEW_CONTROL_VER)"
    fi
else
    fail "DEBIAN/prerm not found"
fi

# 2d. dkms.conf PACKAGE_VERSION (all instances)
DKMS_CONF_COUNT=0
while IFS= read -r -d '' cf; do
    DKMS_CONF_COUNT=$((DKMS_CONF_COUNT + 1))
    DV=$(
        grep -m1 '^PACKAGE_VERSION=' "$cf" \
            | sed 's/^PACKAGE_VERSION="//' \
            | sed 's/"$//'
    )
    REL="${cf#"$VERIFY_DIR"/}"
    if [ "$DV" = "$NEW_CONTROL_VER" ]; then
        pass "$REL PACKAGE_VERSION = $DV"
    else
        fail "$REL PACKAGE_VERSION = $DV (expected $NEW_CONTROL_VER)"
    fi
done < <(
    find "$VERIFY_DIR/usr/src" -name dkms.conf -print0 2>/dev/null
)
if [ "$DKMS_CONF_COUNT" -eq 0 ]; then
    fail "No dkms.conf files found"
fi

# 2e. usr/src/amdgpu-* directory name
NEW_SRC_DIR=$(
    find "$VERIFY_DIR/usr/src" -maxdepth 1 \
        -type d -name 'amdgpu-*' | head -1
)
NEW_DIR_VER="${NEW_SRC_DIR##*/amdgpu-}"
if [ "$NEW_DIR_VER" = "$NEW_CONTROL_VER" ]; then
    pass "Source dir = amdgpu-$NEW_DIR_VER"
else
    fail "Source dir = amdgpu-$NEW_DIR_VER (expected amdgpu-$NEW_CONTROL_VER)"
fi

# 2f. Old version must NOT appear in any checked location
if grep -q "^Version:.*${OLD_CONTROL_VER}\$" \
    "$VERIFY_DIR/DEBIAN/control" 2>/dev/null; then
    fail "Old control version still present"
fi
if grep -q "^CVERSION=${OLD_DKMS_VER}\$" \
    "$VERIFY_DIR/DEBIAN/postinst" 2>/dev/null; then
    fail "Old CVERSION still in postinst"
fi
if grep -q "^VERSION=${OLD_DKMS_VER}\$" \
    "$VERIFY_DIR/DEBIAN/prerm" 2>/dev/null; then
    fail "Old VERSION still in prerm"
fi

echo ""

# --- Step 3: fakeroot install check ---

echo "=== fakeroot install check ==="
fakeroot dpkg-deb -x "$OUT_DEB" "$INSTALL_DIR"

INSTALLED_SRC=$(
    find "$INSTALL_DIR/usr/src" -maxdepth 1 \
        -type d -name 'amdgpu-*' | head -1
)
if [ -z "$INSTALLED_SRC" ]; then
    fail "No amdgpu-* directory after fakeroot install"
else
    INST_VER="${INSTALLED_SRC##*/amdgpu-}"
    if [ "$INST_VER" = "$NEW_CONTROL_VER" ]; then
        pass "Installed source dir = amdgpu-$INST_VER"
    else
        fail "Installed source dir = amdgpu-$INST_VER (expected amdgpu-$NEW_CONTROL_VER)"
    fi

    INST_DKMS="$INSTALLED_SRC/dkms.conf"
    if [ -f "$INST_DKMS" ]; then
        IDV=$(
            grep -m1 '^PACKAGE_VERSION=' "$INST_DKMS" \
                | sed 's/^PACKAGE_VERSION="//' \
                | sed 's/"$//'
        )
        if [ "$IDV" = "$NEW_CONTROL_VER" ]; then
            pass "Installed dkms.conf PACKAGE_VERSION = $IDV"
        else
            fail "Installed dkms.conf PACKAGE_VERSION = $IDV (expected $NEW_CONTROL_VER)"
        fi
    fi

    INST_DKMS_NESTED="$INSTALLED_SRC/amd/dkms/dkms.conf"
    if [ -f "$INST_DKMS_NESTED" ]; then
        IDV=$(
            grep -m1 '^PACKAGE_VERSION=' \
                "$INST_DKMS_NESTED" \
                | sed 's/^PACKAGE_VERSION="//' \
                | sed 's/"$//'
        )
        if [ "$IDV" = "$NEW_CONTROL_VER" ]; then
            pass "Installed amd/dkms/dkms.conf PACKAGE_VERSION = $IDV"
        else
            fail "Installed amd/dkms/dkms.conf PACKAGE_VERSION = $IDV (expected $NEW_CONTROL_VER)"
        fi
    fi
fi

echo ""

# --- Summary ---

echo "=== Results: $PASS passed, $FAIL failed ==="
if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
