#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Delegate to recipies/common/nixl/build-nixl.sh (ROCm + amd-support + AIS).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../../common/nixl/build-nixl.sh" "$@"
