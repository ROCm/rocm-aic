#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Emit the AIC image tag (no image name), e.g.
#   CDNA (gfx9xx):  0.1.0-rocm10.1.0-pytorch2.15-vllm0.34.0-aiter0.1.29.post1-fa0e60e394-lmcache0.5.8rc1-nixl1.5.2-hsasnoop1.6.0-gfx942
#   RDNA (gfx12xx): 0.1.0-rocm10.1.0-pytorch2.15-vllm0.34.0-aiter0.1.29.post1-lmcache0.5.8rc1-nixl1.5.2-hsasnoop1.6.0-gfx1201
#
# The tag is derived in docker/docker-bake.hcl; this is `aic-bake-env.sh --get
# IMAGE_TAG`, kept so existing callers need no change.  A same-named environment
# variable (ROCM_ARCH, ROCM_VERSION, VLLM_REF, ...) overrides the HCL default
# because bake reads variables from the environment.
#
# Usage:  aic-image-tag.sh
#         aic-image-tag.sh [...]   # trailing args accepted for back-compat, ignored
set -euo pipefail

exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/aic-bake-env.sh" --get IMAGE_TAG
