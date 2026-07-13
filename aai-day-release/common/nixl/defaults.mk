# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Pinned NIXL source for rocm-aic ROCm recipes (andyluo7/nixl amd-support + overlays).
NIXL_GIT_URL ?= https://github.com/andyluo7/nixl.git
# Track amd-support; pin SHA for reproducible Docker builds (bump when branch moves).
NIXL_REF ?= amd-support
NIXL_SHA ?= f72aad2cf4da0dff5d710dfcaa8666defa114d78
NIXL_AMD_SUPPORT_SHA ?= $(NIXL_SHA)
