/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include <cassert>
#include <hipfile.h>
#include "ais_backend.h"
#include "ais_utils.h"
#include "common/nixl_log.h"
#include "file/file_utils.h"
#include <unordered_map>
#include <memory>
#include <hip/hip_runtime.h>

#define DEFAULT_BATCH_LIMIT 128
#define DEFAULT_MAX_REQUEST_SIZE (16 * 1024 * 1024)
#define DEFAULT_BATCH_POOL_SIZE 16

nixlAisEngine::nixlAisEngine(const nixlBackendInitParams* init_params)
    : nixlBackendEngine(init_params)
{
    ais_utils = new aisUtil();

    batch_pool_size = DEFAULT_BATCH_POOL_SIZE;
    batch_limit = DEFAULT_BATCH_LIMIT;
    max_request_size = DEFAULT_MAX_REQUEST_SIZE;

    nixl_b_params_t* custom_params = init_params->customParams;
    if (custom_params) {
        if (custom_params->count("batch_pool_size") > 0) {
            try {
                batch_pool_size = std::stoi((*custom_params)["batch_pool_size"]);
            } catch (const std::exception& e) {
                NIXL_ERROR << "AIS: invalid batch_pool_size parameter: " << e.what();
                this->initErr = true;
                return;
            }
        }

        if (custom_params->count("batch_limit") > 0) {
            try {
                batch_limit = std::stoi((*custom_params)["batch_limit"]);
            } catch (const std::exception& e) {
                NIXL_ERROR << "AIS: invalid batch_limit parameter: " << e.what();
                this->initErr = true;
                return;
            }
        }

        if (custom_params->count("max_request_size") > 0) {
            try {
                max_request_size = std::stoul((*custom_params)["max_request_size"]);
            } catch (const std::exception& e) {
                NIXL_ERROR << "AIS: invalid max_request_size parameter: " << e.what();
                this->initErr = true;
                return;
            }
        }
    }

    this->initErr = false;
    if (ais_utils->openAisDriver() == NIXL_ERR_BACKEND) {
        this->initErr = true;
        return;
    }

    for (unsigned int i = 0; i < batch_pool_size; i++) {
        batch_pool.push_back(new nixlAisIOBatch(batch_limit));
    }
}

nixl_status_t nixlAisEngine::registerMem(const nixlBlobDesc &mem,
                                         const nixl_mem_t &nixl_mem,
                                         nixlBackendMD* &out)
{
    nixl_status_t status = NIXL_SUCCESS;
    nixlAisMetadata *md = new nixlAisMetadata();
    md->type = nixl_mem;
    hipError_t error_id;

    switch (nixl_mem) {
        case FILE_SEG: {
            auto it = ais_file_map.find(mem.devId);
            if (it != ais_file_map.end()) {
                md->handle = it->second;
                md->handle.size = mem.len;
                md->handle.metadata = mem.metaInfo;
                break;
            }

            status = ais_utils->registerFileHandle(mem.devId, mem.len,
                                                   mem.metaInfo, md->handle);
            if (status == NIXL_SUCCESS) {
                ais_file_map[mem.devId] = md->handle;
            }
            break;
        }

        case VRAM_SEG: {
            error_id = hipSetDevice(mem.devId);
            if (error_id != hipSuccess) {
                NIXL_ERROR << "AIS: hipSetDevice returned " << hipGetErrorString(error_id)
                          << " for device ID " << mem.devId;
                delete md;
                return NIXL_ERR_BACKEND;
            }
            status = ais_utils->registerBufHandle((void *)mem.addr, mem.len, 0);
            if (status == NIXL_SUCCESS) {
                md->buf.base = (void *)mem.addr;
                md->buf.size = mem.len;
            }
            break;
        }

        case DRAM_SEG: {
            status = ais_utils->registerBufHandle((void *)mem.addr, mem.len, 0);
            if (status == NIXL_SUCCESS) {
                md->buf.base = (void *)mem.addr;
                md->buf.size = mem.len;
            }
            break;
        }

        default:
            status = NIXL_ERR_BACKEND;
            break;
    }

    if (status != NIXL_SUCCESS) {
        delete md;
        return status;
    }

    out = (nixlBackendMD*)md;
    return status;
}

nixl_status_t nixlAisEngine::deregisterMem(nixlBackendMD* meta)
{
    nixlAisMetadata *md = (nixlAisMetadata *)meta;
    if (md->type == FILE_SEG) {
        ais_utils->deregisterFileHandle(md->handle);
        ais_file_map.erase(md->handle.fd);
    } else {
        ais_utils->deregisterBufHandle(md->buf.base);
    }
    delete md;
    return NIXL_SUCCESS;
}

nixl_status_t nixlAisEngine::prepXfer(const nixl_xfer_op_t &operation,
                                      const nixl_meta_dlist_t &local,
                                      const nixl_meta_dlist_t &remote,
                                      const std::string &remote_agent,
                                      nixlBackendReqH* &handle,
                                      const nixl_opt_b_args_t* opt_args) const
{
    nixlAisBackendReqH* ais_handle = new nixlAisBackendReqH();
    size_t buf_cnt = local.descCount();
    size_t file_cnt = remote.descCount();

    if ((buf_cnt != file_cnt) ||
        ((operation != NIXL_READ) && (operation != NIXL_WRITE))) {
        NIXL_ERROR << "AIS: error in count or operation selection";
        delete ais_handle;
        return NIXL_ERR_INVALID_PARAM;
    }

    if ((remote.getType() != FILE_SEG) && (local.getType() != FILE_SEG)) {
        NIXL_ERROR << "AIS: only support I/O between memory (DRAM/VRAM) and file type";
        delete ais_handle;
        return NIXL_ERR_INVALID_PARAM;
    }

    ais_handle->request_list.clear();

    bool is_local_file = (local.getType() == FILE_SEG);

    for (size_t i = 0; i < buf_cnt; i++) {
        void* base_addr;
        size_t total_size;
        size_t base_offset;
        aisFileHandle fh;

        if (is_local_file) {
            base_addr = (void*)remote[i].addr;
            if (!base_addr) {
                delete ais_handle;
                return NIXL_ERR_INVALID_PARAM;
            }
            total_size = remote[i].len;
            base_offset = (size_t)local[i].addr;

            auto it = ais_file_map.find(local[i].devId);
            if (it == ais_file_map.end()) {
                NIXL_ERROR << "AIS: file handle not found";
                delete ais_handle;
                return NIXL_ERR_NOT_FOUND;
            }
            fh = it->second;
        } else {
            base_addr = (void*)local[i].addr;
            if (!base_addr) {
                delete ais_handle;
                return NIXL_ERR_INVALID_PARAM;
            }
            total_size = local[i].len;
            base_offset = (size_t)remote[i].addr;

            auto it = ais_file_map.find(remote[i].devId);
            if (it == ais_file_map.end()) {
                NIXL_ERROR << "AIS: file handle not found";
                delete ais_handle;
                return NIXL_ERR_NOT_FOUND;
            }
            fh = it->second;
        }

        size_t remaining_size = total_size;
        size_t current_offset = 0;

        while (remaining_size > 0) {
            size_t request_size = std::min(remaining_size,
                                       (size_t)max_request_size);

            AisTransferRequestH req;
            req.addr = (char*)base_addr + current_offset;
            req.size = request_size;
            req.file_offset = base_offset + current_offset;
            req.fh = fh.hip_fhandle;
            req.op = (operation == NIXL_READ) ? hipFileBatchRead : hipFileBatchWrite;

            ais_handle->request_list.push_back(req);

            remaining_size -= request_size;
            current_offset += request_size;
        }
    }

    if (ais_handle->request_list.empty()) {
        delete ais_handle;
        return NIXL_ERR_INVALID_PARAM;
    }

    ais_handle->needs_prep = false;
    handle = ais_handle;
    return NIXL_SUCCESS;
}

nixlAisIOBatch* nixlAisEngine::getBatchFromPool(unsigned int size) const {
    const std::lock_guard<std::mutex> lock(batch_pool_lock);
    if (!batch_pool.empty()) {
        nixlAisIOBatch* batch = batch_pool.back();
        batch_pool.pop_back();
        batch->reset();
        return batch;
    }
    return nullptr;
}

void nixlAisEngine::returnBatchToPool(nixlAisIOBatch* batch) const {
    const std::lock_guard<std::mutex> lock(batch_pool_lock);
    batch_pool.push_back(batch);
}

nixl_status_t nixlAisEngine::postXfer(const nixl_xfer_op_t &operation,
                                      const nixl_meta_dlist_t &local,
                                      const nixl_meta_dlist_t &remote,
                                      const std::string &remote_agent,
                                      nixlBackendReqH* &handle,
                                      const nixl_opt_b_args_t* opt_args) const
{
    nixlAisBackendReqH* ais_handle = (nixlAisBackendReqH*)handle;

    if (ais_handle->request_list.empty()) {
        NIXL_ERROR << "AIS: empty request list";
        return NIXL_ERR_INVALID_PARAM;
    }

    const auto& request_list = ais_handle->request_list;
    size_t current_req = 0;

    while (current_req < request_list.size()) {
        size_t batch_size = std::min(request_list.size() - current_req,
                                     (size_t)batch_limit);
        nixl_status_t status = createAndSubmitBatch(request_list, current_req,
                                                    batch_size, ais_handle->batch_io_list);

        if (status != NIXL_SUCCESS) {
            for (auto* batch : ais_handle->batch_io_list) {
                batch->cancelBatch();
                returnBatchToPool(batch);
            }
            ais_handle->batch_io_list.clear();
            return status;
        }
        current_req += batch_size;
    }

    return NIXL_IN_PROG;
}

nixl_status_t nixlAisEngine::createAndSubmitBatch(const std::vector<AisTransferRequestH>& requests,
                                                  size_t start_idx, size_t batch_size,
                                                  std::vector<nixlAisIOBatch*>& batch_list) const
{
    nixlAisIOBatch* batch = getBatchFromPool(batch_size);
    if (!batch) {
        NIXL_ERROR << "AIS: batch pool exhausted";
        return NIXL_ERR_BACKEND;
    }

    for (size_t i = 0; i < batch_size; i++) {
        const auto& req = requests[start_idx + i];
        if (!req.addr || !req.fh) {
            returnBatchToPool(batch);
            return NIXL_ERR_INVALID_PARAM;
        }

        nixl_status_t status = batch->addToBatch(req.fh, req.addr, req.size,
                                               req.file_offset, 0, req.op);
        if (status != NIXL_SUCCESS) {
            returnBatchToPool(batch);
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    nixl_status_t status = batch->submitBatch(0);
    if (status != NIXL_SUCCESS) {
        returnBatchToPool(batch);
        return NIXL_ERR_BACKEND;
    }

    batch_list.push_back(batch);
    return NIXL_SUCCESS;
}

nixl_status_t nixlAisEngine::checkXfer(nixlBackendReqH* handle) const
{
    nixlAisBackendReqH *ais_handle = (nixlAisBackendReqH *)handle;

    if (ais_handle->batch_io_list.empty()) {
        ais_handle->needs_prep = true;
        return NIXL_SUCCESS;
    }

    nixl_status_t status = NIXL_SUCCESS;
    for (auto* batch : ais_handle->batch_io_list) {
        status = batch->checkStatus();

        if (status == NIXL_IN_PROG) {
            return status;
        }

        if (status < 0) {
            batch->cancelBatch();
        }
        returnBatchToPool(batch);
    }

    ais_handle->batch_io_list.clear();
    ais_handle->needs_prep = true;
    return status;
}

nixl_status_t nixlAisEngine::releaseReqH(nixlBackendReqH* handle) const
{
    nixlAisBackendReqH *ais_handle = (nixlAisBackendReqH *)handle;
    delete ais_handle;
    return NIXL_SUCCESS;
}

nixlAisEngine::~nixlAisEngine() {
    for (auto* batch : batch_pool) {
        if (batch) {
            delete batch;
        }
    }
    batch_pool.clear();

    if (ais_utils) {
        ais_utils->closeAisDriver();
        delete ais_utils;
    }
}

nixl_status_t
nixlAisEngine::queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const {
    std::vector<nixl_blob_t> metadata(descs.descCount());
    for (int i = 0; i < descs.descCount(); ++i)
        metadata[i] = descs[i].metaInfo;

    return nixl::queryFileInfoList(metadata, resp);
}
