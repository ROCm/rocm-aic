/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef __AIS_UTILS_H
#define __AIS_UTILS_H

#include <fcntl.h>
#include <unistd.h>
#include <nixl.h>
#include <hipfile.h>

class aisFileHandle {
    public:
        int fd;
        size_t size;
        std::string metadata;
        hipFileHandle_t hip_fhandle;
};

class aisMemBuf {
    public:
        void *base;
        size_t size;
};

class nixlAisIOBatch {
    public:
        nixlAisIOBatch(unsigned int size);
        ~nixlAisIOBatch();

        nixl_status_t addToBatch(hipFileHandle_t fh, void *buffer,
                                size_t size, size_t file_offset,
                                size_t ptr_offset, hipFileOpcode_t type);
        nixl_status_t submitBatch(int flags);
        nixl_status_t checkStatus();
        nixl_status_t cancelBatch();
        void reset();

    private:
        hipFileBatchHandle_t batch_handle;
        hipFileIOEvents_t *io_batch_events = nullptr;
        hipFileIOParams_t *io_batch_params = nullptr;
        hipFileError_t init_err = {hipFileSuccess};
        unsigned int max_reqs = 0;
        unsigned int batch_size = 0;
        unsigned int entries_completed = 0;
        nixl_status_t current_status = NIXL_ERR_NOT_POSTED;
};

class aisUtil {
    public:
        aisUtil() {}
        ~aisUtil() {}
        nixl_status_t registerFileHandle(int fd, size_t size,
                                       std::string metaInfo,
                                       aisFileHandle& handle);
        nixl_status_t registerBufHandle(void *ptr, size_t size, int flags);
        void deregisterFileHandle(aisFileHandle& handle);
        nixl_status_t deregisterBufHandle(void *ptr);
        nixl_status_t openAisDriver();
        void closeAisDriver();
};
#endif
