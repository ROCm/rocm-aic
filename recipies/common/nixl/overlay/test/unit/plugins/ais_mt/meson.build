# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

if cuda_dep.found() == false or not use_rocm
    subdir_done()
endif

nixl_ais_mt_app = executable('nixl_ais_mt_test', 'nixl_ais_mt_test.cpp',
                              dependencies: [nixl_dep, nixl_infra, cuda_dep, nixl_common_dep],
                              include_directories: [nixl_inc_dirs, utils_inc_dirs],
                              install: true)
