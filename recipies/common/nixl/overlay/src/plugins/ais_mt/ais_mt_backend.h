/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef __AIS_MT_BACKEND_H
#define __AIS_MT_BACKEND_H

#include <nixl.h>
#include <nixl_types.h>
#include <backend/backend_engine.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <hipfile.h>
#include "ais_mt_utils.h"
#include "taskflow/core/executor.hpp"

class nixlAisMtEngine : public nixlBackendEngine {
public:
    nixlAisMtEngine (const nixlBackendInitParams *init_params);
    ~nixlAisMtEngine() = default;

    nixlAisMtEngine (const nixlAisMtEngine &) = delete;
    nixlAisMtEngine &
    operator= (const nixlAisMtEngine &) = delete;

    bool
    supportsNotif() const override {
        return false;
    }
    bool
    supportsRemote() const override {
        return false;
    }
    bool
    supportsLocal() const override {
        return true;
    }

    nixl_mem_list_t
    getSupportedMems() const override {
        return {DRAM_SEG, VRAM_SEG, FILE_SEG};
    }

    nixl_status_t
    connect (const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    disconnect (const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    loadLocalMD (nixlBackendMD *input, nixlBackendMD *&output) override {
        output = input;
        return NIXL_SUCCESS;
    }

    nixl_status_t
    unloadMD (nixlBackendMD *input) override {
        return NIXL_SUCCESS;
    }
    nixl_status_t
    registerMem (const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;
    nixl_status_t
    deregisterMem (nixlBackendMD *meta) override;

    nixl_status_t
    prepXfer (const nixl_xfer_op_t &operation,
              const nixl_meta_dlist_t &local,
              const nixl_meta_dlist_t &remote,
              const std::string &remote_agent,
              nixlBackendReqH *&handle,
              const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    postXfer (const nixl_xfer_op_t &operation,
              const nixl_meta_dlist_t &local,
              const nixl_meta_dlist_t &remote,
              const std::string &remote_agent,
              nixlBackendReqH *&handle,
              const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    checkXfer (nixlBackendReqH *handle) const override;
    nixl_status_t
    releaseReqH (nixlBackendReqH *handle) const override;

    nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const override;

private:
    aisMtUtil ais_mt_utils_;
    std::unordered_map<int, std::weak_ptr<aisMtFileHandle>> ais_mt_file_map_;
    size_t thread_count_;
    std::unique_ptr<tf::Executor> executor_;
};
#endif
