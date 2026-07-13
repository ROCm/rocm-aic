# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

use_rocm = get_option('use_rocm')
if use_rocm != ''
    ais_lib_path = use_rocm + '/lib'
    ais_inc_path = use_rocm + '/include'
    hipfile_dep = declare_dependency(
        link_args: ['-L' + ais_lib_path, '-lhipfile'],
        include_directories: include_directories(ais_inc_path))
else
    hipfile_dep = declare_dependency()
    subdir_done()
endif

if 'AIS' in static_plugins
    ais_backend_lib = static_library('AIS',
        'ais_utils.cpp', 'ais_utils.h',
        'ais_backend.cpp', 'ais_backend.h',
        'ais_plugin.cpp',
        dependencies: [nixl_infra, nixl_common_dep, cuda_dep, hipfile_dep, file_utils_interface],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: false,
        cpp_args: compile_flags,
        name_prefix: 'libplugin_')
else
    ais_backend_lib = shared_library('AIS',
        'ais_utils.cpp', 'ais_utils.h',
        'ais_backend.cpp', 'ais_backend.h',
        'ais_plugin.cpp',
        dependencies: [nixl_infra, nixl_common_dep, cuda_dep, hipfile_dep, file_utils_interface],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: true,
        cpp_args: ['-fPIC'],
        name_prefix: 'libplugin_',
        install_dir: plugin_install_dir,
        install_rpath: '$ORIGIN/..')
    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
            'echo "AIS=' + ais_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true
        )
    endif
endif

ais_backend_interface = declare_dependency(link_with: ais_backend_lib)
