#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Print the gfx arch of the node that `test` / `tiny-test` will land on.
#
# The -fast targets build and test a SINGLE-arch image, and the tarball name is
# derived from that arch -- so the arch has to be known on the submitting host,
# before any job runs.  Hardcoding it (the old AIC_FAST_ARCH ?= gfx950) breaks
# as soon as the test node is not an MI355X: `make dist-build-fast` writes a
# gfx950 tarball and `make tiny-test-fast` looks for it on a gfx942 node.
#
# Resolution order (first hit wins):
#   1. AIC_TEST_ARCH        explicit override, used verbatim
#   2. AIC_SPUR_CLUSTER=1   -> gfx950 (every SPUR node is MI355X)
#   3. AIC_TEST_NODE        -> the GFX* feature of that node   (sinfo)
#   4. AIC_TEST_CONSTRAINT  -> a GFX* term in the feature expr (no sinfo needed)
#   5. AIC_TEST_PARTITION   -> the GFX* feature of its nodes   (sinfo)
#   6. local GPU            -> rocm_agent_enumerator, for non-Slurm hosts
#   7. gfx950               -> last-resort default (warns on stderr)
#
# Usage:  .slurm/aic-test-arch.sh
set -uo pipefail

# Slurm feature labels are upper-case and sometimes carry a form suffix that is
# not part of the arch (GFX1100W, GFX1100P, GFX1101V).  Lower-case, then strip
# trailing letters until the result is a real arch -- but only when needed, so
# genuinely letter-suffixed archs (gfx90a) survive.
_KNOWN_ARCHS="gfx906 gfx908 gfx90a gfx940 gfx941 gfx942 gfx950 \
gfx1030 gfx1100 gfx1101 gfx1102 gfx1150 gfx1151 gfx1200 gfx1201"

_normalize() {
    local a; a="$(printf '%s' "${1}" | tr '[:upper:]' '[:lower:]')"
    while [[ -n "${a}" ]]; do
        case " ${_KNOWN_ARCHS} " in
            *" ${a} "*) printf '%s' "${a}"; return 0 ;;
        esac
        # Not a known arch: drop a trailing letter and retry (gfx1100w -> gfx1100).
        [[ "${a}" =~ [a-z]$ ]] || return 1
        a="${a%?}"
    done
    return 1
}

# Pull the first GFX* token out of a comma/&/|-separated feature string.
_arch_from_features() {
    local feats="${1}" tok
    for tok in $(printf '%s' "${feats}" | tr ',&|()!' '\n\n\n\n\n\n\n'); do
        [[ "${tok}" == GFX* || "${tok}" == gfx* ]] || continue
        _normalize "${tok}" && return 0
    done
    return 1
}

_arch=""

# 1. explicit override
if [[ -n "${AIC_TEST_ARCH:-}" ]]; then
    printf '%s\n' "${AIC_TEST_ARCH}"
    exit 0
fi

# 2. SPUR is homogeneous MI355X
if [[ "${AIC_SPUR_CLUSTER:-0}" == "1" ]]; then
    printf 'gfx950\n'
    exit 0
fi

if command -v sinfo >/dev/null 2>&1; then
    # 3. an explicitly pinned node is the most specific signal
    if [[ -z "${_arch}" && -n "${AIC_TEST_NODE:-}" ]]; then
        _feat="$(sinfo -h -n "${AIC_TEST_NODE}" -o '%f' 2>/dev/null | head -1)"
        [[ -n "${_feat}" ]] && _arch="$(_arch_from_features "${_feat}" || true)"
    fi

    # 4. the constraint often names the arch outright -- free, no query
    if [[ -z "${_arch}" && -n "${AIC_TEST_CONSTRAINT:-}" ]]; then
        _arch="$(_arch_from_features "${AIC_TEST_CONSTRAINT}" || true)"
    fi

    # 5. otherwise ask the partition.  Take the most common arch among its nodes
    #    so a mixed partition still yields the majority target.
    if [[ -z "${_arch}" ]]; then
        _part="${AIC_TEST_PARTITION:-${AIC_BUILD_PARTITION:-defq}}"
        _arch="$(
            sinfo -h -p "${_part}" -o '%f' 2>/dev/null |
            tr ',' '\n' | grep -iE '^gfx' | sort | uniq -c | sort -rn |
            awk 'NR==1{print $2}'
        )"
        [[ -n "${_arch}" ]] && _arch="$(_normalize "${_arch}" || true)"
    fi
fi

# 6. No Slurm at all: the job would run here, so this host's GPU IS the target.
#    Guarded on sinfo being absent -- on a submit host the local GPU (login nodes
#    often have one) says nothing about the node the test will actually land on.
if [[ -z "${_arch}" ]] && ! command -v sinfo >/dev/null 2>&1 &&
        command -v rocm_agent_enumerator >/dev/null 2>&1; then
    _arch="$(rocm_agent_enumerator 2>/dev/null | grep -m1 -E '^gfx' || true)"
    [[ -n "${_arch}" ]] && _arch="$(_normalize "${_arch}" || true)"
fi

# 7. last resort
if [[ -z "${_arch}" ]]; then
    printf 'aic-test-arch: could not detect the test-node arch; defaulting to gfx950 (set AIC_TEST_ARCH to override)\n' >&2
    _arch="gfx950"
fi

printf '%s\n' "${_arch}"
