/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include <nixl.h>
#include <nixl_types.h>
#include <backend/backend_engine.h>
#include <hip/hip_runtime.h>
#include <hipfile.h>
#include <thread>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <exception>
#include <cstring>
#include <variant>
#include <future>
#include <atomic>
#include "common/nixl_log.h"
#include "ais_mt_backend.h"
#include "ais_mt_utils.h"
#include "file/file_utils.h"
#include <taskflow/taskflow.hpp>
#include <unordered_map>

namespace {
const size_t default_thread_count = std::max (1u, std::thread::hardware_concurrency() / 2);

struct FileSegData {
    std::shared_ptr<aisMtFileHandle> handle;

    FileSegData(std::shared_ptr<aisMtFileHandle> h) : handle(std::move(h)) {}
};

struct MemSegData {
    std::unique_ptr<aisMtMemBuf> buf;
    MemSegData (void *addr, size_t size, int flags)
        : buf (std::make_unique<aisMtMemBuf> (addr, size, flags)) {}
};

struct AisMtTransferRequestH {
    AisMtTransferRequestH (void *a,
                           size_t s,
                           size_t offset,
                           hipFileHandle_t handle,
                           hipFileOpcode_t operation)
        : addr{a},
          size{s},
          file_offset{offset},
          fh{handle},
          op{operation} {}

    void *addr;
    size_t size;
    size_t file_offset;
    hipFileHandle_t fh;
    hipFileOpcode_t op;
};

class nixlAisMtMetadata : public nixlBackendMD {
public:
    explicit nixlAisMtMetadata (std::shared_ptr<aisMtFileHandle> file_handle)
        : nixlBackendMD (true),
          data_ (FileSegData{std::move (file_handle)}) {}

    explicit nixlAisMtMetadata (void *addr, size_t size, int flags)
        : nixlBackendMD (true),
          data_ (MemSegData{addr, size, flags}) {}

    ~nixlAisMtMetadata() = default;

    nixlAisMtMetadata (const nixlAisMtMetadata &) = delete;
    nixlAisMtMetadata &
    operator= (const nixlAisMtMetadata &) = delete;

    nixlAisMtMetadata (nixlAisMtMetadata &&) = default;
    nixlAisMtMetadata &
    operator= (nixlAisMtMetadata &&) = default;

    std::variant<FileSegData, MemSegData> data_;
};

class nixlAisMtBackendReqH : public nixlBackendReqH {
public:
    ~nixlAisMtBackendReqH();

    std::vector<AisMtTransferRequestH> request_list;
    tf::Taskflow taskflow;
    std::future<void> running_transfer;
    std::atomic<nixl_status_t> overall_status;
};

size_t
getThreadCount (const nixlBackendInitParams *init_params) {
    size_t thread_count = default_thread_count;

    nixl_b_params_t *custom_params = init_params->customParams;
    if (custom_params) {
        if (custom_params->count ("thread_count") > 0) {
            try {
                size_t tcount = std::stoul ((*custom_params)["thread_count"]);
                if (tcount != 0) {
                    thread_count = tcount;
                }
            }
            catch (const std::exception &e) {
                throw std::runtime_error ("AIS_MT: invalid thread_count parameter: " +
                                          std::string (e.what()));
            }
        }
    }
    return thread_count;
}

void
runHipFileOp (AisMtTransferRequestH *req, std::atomic<nixl_status_t> *overall_status) {
    ssize_t nbytes = 0;
    if (req->op == hipFileBatchRead) {
        nbytes = hipFileRead (req->fh, req->addr, req->size, req->file_offset, 0);
        if (nbytes < 0) {
            NIXL_ERROR << "AIS_MT: hipFileRead failed: " << strerror (errno);
            overall_status->store (NIXL_ERR_BACKEND);
            return;
        }
    } else if (req->op == hipFileBatchWrite) {
        nbytes = hipFileWrite (req->fh, req->addr, req->size, req->file_offset, 0);
        if (nbytes < 0) {
            NIXL_ERROR << "AIS_MT: hipFileWrite failed: " << strerror (errno);
            overall_status->store (NIXL_ERR_BACKEND);
            return;
        }
    } else {
        overall_status->store (NIXL_ERR_INVALID_PARAM);
        return;
    }

    if ((size_t)nbytes != req->size) {
        NIXL_ERROR << "AIS_MT: error: short " << ((req->op == hipFileBatchRead) ? "read: " : "write: ")
                   << nbytes << " out of " << req->size << " bytes - address=" << req->addr;
        overall_status->store (NIXL_ERR_BACKEND);
        return;
    }
}

nixl_status_t
extractTransferParams (const nixlMetaDesc &mem_desc,
                       const nixlMetaDesc &file_desc,
                       const std::unordered_map<int, std::weak_ptr<aisMtFileHandle>> &file_map,
                       void *&base_addr,
                       size_t &total_size,
                       size_t &base_offset,
                       hipFileHandle_t &hip_fhandle) {
    base_addr = (void *)mem_desc.addr;
    total_size = mem_desc.len;
    base_offset = (size_t)file_desc.addr;

    auto it = file_map.find (file_desc.devId);
    if (it == file_map.end()) {
        NIXL_ERROR << "AIS_MT: error: file metadata not found";
        return NIXL_ERR_NOT_FOUND;
    }

    auto handle = it->second.lock();
    NIXL_ASSERT (handle);
    hip_fhandle = handle->hip_fhandle;
    return NIXL_SUCCESS;
}
} // namespace

nixlAisMtBackendReqH::~nixlAisMtBackendReqH() {
    if (running_transfer.valid()) {
        running_transfer.wait();
    }
}

nixlAisMtEngine::nixlAisMtEngine (const nixlBackendInitParams *init_params)
    : nixlBackendEngine (init_params),
      ais_mt_utils_(),
      thread_count_ (getThreadCount (init_params)),
      executor_ (std::make_unique<tf::Executor> (thread_count_)) {
    NIXL_DEBUG << "AIS_MT: thread count=" << thread_count_;
}

nixl_status_t
nixlAisMtEngine::registerMem (const nixlBlobDesc &mem,
                              const nixl_mem_t &nixl_mem,
                              nixlBackendMD *&out) {
    switch (nixl_mem) {
    case FILE_SEG: {
        auto it = ais_mt_file_map_.find (mem.devId);
        std::shared_ptr<aisMtFileHandle> handle;
        if (it != ais_mt_file_map_.end()) {
            handle = it->second.lock();
            if (handle) {
                out = new nixlAisMtMetadata (handle);
                return NIXL_SUCCESS;
            }
            ais_mt_file_map_.erase (it);
        }

        try {
            handle = std::make_shared<aisMtFileHandle> (mem.devId);
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "AIS_MT: failed to create file handle: " << e.what();
            return NIXL_ERR_BACKEND;
        }
        ais_mt_file_map_[mem.devId] = handle;
        out = new nixlAisMtMetadata (handle);
        return NIXL_SUCCESS;
    }

    case VRAM_SEG: {
        const hipError_t error_id = hipSetDevice (mem.devId);
        if (error_id != hipSuccess) {
            NIXL_ERROR << "AIS_MT: error: hipSetDevice returned " << hipGetErrorString (error_id)
                       << " for device ID " << mem.devId;
            return NIXL_ERR_BACKEND;
        }
        [[fallthrough]];
    }

    case DRAM_SEG: {
        try {
            out = new nixlAisMtMetadata ((void *)mem.addr, mem.len, 0);
            return NIXL_SUCCESS;
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "AIS_MT: failed to create memory buffer: " << e.what();
            return NIXL_ERR_BACKEND;
        }
    }

    default:
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlAisMtEngine::deregisterMem (nixlBackendMD *meta) {
    std::unique_ptr<nixlAisMtMetadata> md ((nixlAisMtMetadata *)meta);

    if (auto *file_data = std::get_if<FileSegData> (&md->data_)) {
        if (file_data->handle) {
            int key = file_data->handle->fd;
            md.reset();

            auto it = ais_mt_file_map_.find (key);
            if (it != ais_mt_file_map_.end() && it->second.expired()) {
                ais_mt_file_map_.erase (it);
            }
        }
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlAisMtEngine::prepXfer (const nixl_xfer_op_t &operation,
                           const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           const std::string &remote_agent,
                           nixlBackendReqH *&handle,
                           const nixl_opt_b_args_t *opt_args) const {
    auto ais_mt_handle = std::make_unique<nixlAisMtBackendReqH>();
    size_t buf_cnt = local.descCount();
    size_t file_cnt = remote.descCount();

    if ((buf_cnt != file_cnt) || ((operation != NIXL_READ) && (operation != NIXL_WRITE))) {
        NIXL_ERROR << "AIS_MT: error: incorrect count or operation selection";
        return NIXL_ERR_INVALID_PARAM;
    }

    if ((remote.getType() != FILE_SEG) && (local.getType() != FILE_SEG)) {
        NIXL_ERROR << "AIS_MT: error: backend only supports I/O between memory (DRAM/VRAM_SEG) and "
                      "files (FILE_SEG)";
        return NIXL_ERR_INVALID_PARAM;
    }

    ais_mt_handle->request_list.clear();
    bool is_local_file = (local.getType() == FILE_SEG);
    for (size_t i = 0; i < buf_cnt; i++) {
        void *base_addr;
        size_t total_size;
        size_t base_offset;
        hipFileHandle_t hip_fhandle;

        nixl_status_t param_status;
        if (is_local_file) {
            param_status = extractTransferParams (remote[i],
                                                  local[i],
                                                  ais_mt_file_map_,
                                                  base_addr,
                                                  total_size,
                                                  base_offset,
                                                  hip_fhandle);
        } else {
            param_status = extractTransferParams (local[i],
                                                  remote[i],
                                                  ais_mt_file_map_,
                                                  base_addr,
                                                  total_size,
                                                  base_offset,
                                                  hip_fhandle);
        }

        if (param_status != NIXL_SUCCESS) {
            return param_status;
        }

        ais_mt_handle->request_list.emplace_back (base_addr,
                                                  total_size,
                                                  base_offset,
                                                  hip_fhandle,
                                                  (operation == NIXL_READ) ? hipFileBatchRead :
                                                                             hipFileBatchWrite);
    }

    if (ais_mt_handle->request_list.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    for (AisMtTransferRequestH &req : ais_mt_handle->request_list) {
        AisMtTransferRequestH *captured_req = &req;
        ais_mt_handle->taskflow.emplace (
            [captured_req, overall_status = &ais_mt_handle->overall_status]() {
                runHipFileOp (captured_req, overall_status);
            });
    }

    handle = ais_mt_handle.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAisMtEngine::postXfer (const nixl_xfer_op_t &operation,
                           const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           const std::string &remote_agent,
                           nixlBackendReqH *&handle,
                           const nixl_opt_b_args_t *opt_args) const {
    nixlAisMtBackendReqH *ais_mt_handle = (nixlAisMtBackendReqH *)handle;

    ais_mt_handle->overall_status.store (NIXL_SUCCESS);
    ais_mt_handle->running_transfer = executor_->run (ais_mt_handle->taskflow);
    return NIXL_IN_PROG;
}

nixl_status_t
nixlAisMtEngine::checkXfer (nixlBackendReqH *handle) const {
    nixlAisMtBackendReqH *ais_mt_handle = (nixlAisMtBackendReqH *)handle;
    if (ais_mt_handle->running_transfer.wait_for (nixlTime::seconds (0)) !=
        std::future_status::ready) {
        return NIXL_IN_PROG;
    }
    ais_mt_handle->running_transfer.get();
    return ais_mt_handle->overall_status.load();
}

nixl_status_t
nixlAisMtEngine::releaseReqH (nixlBackendReqH *handle) const {
    std::unique_ptr<nixlAisMtBackendReqH> ais_mt_handle ((nixlAisMtBackendReqH *)handle);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAisMtEngine::queryMem(const nixl_reg_dlist_t &descs,
                          std::vector<nixl_query_resp_t> &resp) const {
    std::vector<nixl_blob_t> metadata(descs.descCount());
    for (int i = 0; i < descs.descCount(); ++i)
        metadata[i] = descs[i].metaInfo;

    return nixl::queryFileInfoList(metadata, resp);
}
