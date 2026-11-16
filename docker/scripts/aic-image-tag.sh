#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Emit the AIC image tag (no image name), e.g.
#   CDNA (gfx9xx):  0.1.0-rocm7.14.1-pytorch2.13-vllm0.29.0-aiter0.1.22.post1-fa2.8.3.post1-lmcache0.5.6-nixl1.4.1-hsasnoop1.2.0-gfx942
#   RDNA (gfx12xx): 0.1.0-rocm7.14.1-pytorch2.13-vllm0.29.0-aiter0.1.22.post1-lmcache0.5.6-nixl1.4.1-hsasnoop1.2.0-gfx1250
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
