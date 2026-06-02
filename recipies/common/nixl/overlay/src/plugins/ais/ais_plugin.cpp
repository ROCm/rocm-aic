/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include "backend/backend_plugin.h"
#include "ais_backend.h"

using ais_plugin_t = nixlBackendPluginCreator<nixlAisEngine>;

#ifdef STATIC_PLUGIN_AIS
nixlBackendPlugin *
createStaticAISPlugin() {
    return ais_plugin_t::create(
        NIXL_PLUGIN_API_VERSION, "AIS", "0.1.0", {}, {DRAM_SEG, VRAM_SEG, FILE_SEG});
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return ais_plugin_t::create(
        NIXL_PLUGIN_API_VERSION, "AIS", "0.1.0", {}, {DRAM_SEG, VRAM_SEG, FILE_SEG});
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
