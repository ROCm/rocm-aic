/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef __AIS_BACKEND_H
#define __AIS_BACKEND_H

#include <nixl.h>
#include <nixl_types.h>
#include <hip/hip_runtime.h>
#include <unistd.h>
#include <fcntl.h>
#include <list>
#include <vector>
#include <mutex>
#include "ais_utils.h"
#include "backend/backend_engine.h"

class nixlAisMetadata : public nixlBackendMD {
    public:
        aisFileHandle handle;
        aisMemBuf buf;
        nixl_mem_t type;

        nixlAisMetadata() : nixlBackendMD(true) { }
        ~nixlAisMetadata() { }
};

class AisTransferRequestH {
    public:
        void*             addr;
        size_t            size;
        size_t            file_offset;
        hipFileHandle_t   fh;
        hipFileOpcode_t   op;

        AisTransferRequestH() {
            addr = nullptr;
            size = 0;
            file_offset = 0;
            fh = nullptr;
            op = hipFileBatchRead;
        }

        AisTransferRequestH(void* a, size_t s, size_t offset,
                            hipFileHandle_t handle, hipFileOpcode_t operation) {
            addr = a;
            size = s;
            file_offset = offset;
            fh = handle;
            op = operation;
        }
};

class nixlAisBackendReqH : public nixlBackendReqH {
    public:
        std::vector<AisTransferRequestH> request_list;
        std::vector<nixlAisIOBatch*> batch_io_list;
        bool needs_prep;

        nixlAisBackendReqH() {
            needs_prep = true;
        }
        ~nixlAisBackendReqH() {
            for (auto* batch : batch_io_list) {
                delete batch;
            }
            batch_io_list.clear();
        }
};

class nixlAisEngine : public nixlBackendEngine {
    private:
        aisUtil *ais_utils;
        std::unordered_map<int, aisFileHandle> ais_file_map;

        mutable std::mutex batch_pool_lock;
        mutable std::list<nixlAisIOBatch*> batch_pool;
        unsigned int batch_pool_size;
        unsigned int batch_limit;
        unsigned int max_request_size;

        nixlAisIOBatch* getBatchFromPool(unsigned int size) const;
        void returnBatchToPool(nixlAisIOBatch* batch) const;
        nixl_status_t createAndSubmitBatch(const std::vector<AisTransferRequestH>& requests,
                                           size_t start_idx, size_t batch_size,
                                           std::vector<nixlAisIOBatch*>& batch_list) const;

    public:
        nixlAisEngine(const nixlBackendInitParams* init_params);
        ~nixlAisEngine();

        bool supportsNotif() const {
            return false;
        }
        bool supportsRemote() const {
            return false;
        }
        bool supportsLocal() const {
            return true;
        }

        nixl_mem_list_t getSupportedMems() const {
            nixl_mem_list_t mems;
            mems.push_back(DRAM_SEG);
            mems.push_back(VRAM_SEG);
            mems.push_back(FILE_SEG);
            return mems;
        }

        nixl_status_t connect(const std::string &remote_agent) {
            return NIXL_SUCCESS;
        }

        nixl_status_t disconnect(const std::string &remote_agent) {
            return NIXL_SUCCESS;
        }

        nixl_status_t loadLocalMD(nixlBackendMD* input,
                                 nixlBackendMD* &output) {
            output = input;
            return NIXL_SUCCESS;
        }

        nixl_status_t unloadMD(nixlBackendMD* input) {
            return NIXL_SUCCESS;
        }
        nixl_status_t registerMem(const nixlBlobDesc &mem,
                                 const nixl_mem_t &nixl_mem,
                                 nixlBackendMD* &out);
        nixl_status_t deregisterMem(nixlBackendMD *meta);

        nixl_status_t prepXfer(const nixl_xfer_op_t &operation,
                              const nixl_meta_dlist_t &local,
                              const nixl_meta_dlist_t &remote,
                              const std::string &remote_agent,
                              nixlBackendReqH* &handle,
                              const nixl_opt_b_args_t* opt_args=nullptr) const;

        nixl_status_t postXfer(const nixl_xfer_op_t &operation,
                              const nixl_meta_dlist_t &local,
                              const nixl_meta_dlist_t &remote,
                              const std::string &remote_agent,
                              nixlBackendReqH* &handle,
                              const nixl_opt_b_args_t* opt_args=nullptr) const;

        nixl_status_t checkXfer(nixlBackendReqH* handle) const;
        nixl_status_t releaseReqH(nixlBackendReqH* handle) const;

        nixl_status_t
        queryMem(const nixl_reg_dlist_t &descs,
                 std::vector<nixl_query_resp_t> &resp) const override;
};
#endif
