# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Pinned NIXL source for rocm-aic recipes (andyluo7/nixl amd-support + AIS overlay).
NIXL_GIT_URL ?= https://github.com/andyluo7/nixl.git
NIXL_REF ?= amd-support
# amd-support HEAD at plan time; overlay adds AIS/AIS_MT at build time.
NIXL_AMD_SUPPORT_SHA ?= 3340b20b10d916fc3e12e20c57c53dcbc4204ed3
