/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include "ais_utils.h"
#include "common/nixl_log.h"

nixl_status_t aisUtil::registerFileHandle(int fd,
                                          size_t size,
                                          std::string metaInfo,
                                          aisFileHandle& ais_handle)
{
    hipFileError_t status;
    hipFileDescr_t descr = {};
    hipFileHandle_t handle;

    descr.handle.fd = fd;
    descr.type = hipFileHandleTypeOpaqueFD;

    status = hipFileHandleRegister(&handle, &descr);
    if (status.err != hipFileSuccess) {
        NIXL_ERROR << "AIS: file register error";
        return NIXL_ERR_BACKEND;
    }

    ais_handle.hip_fhandle = handle;
    ais_handle.fd = fd;
    ais_handle.size = size;
    ais_handle.metadata = metaInfo;

    return NIXL_SUCCESS;
}

nixl_status_t aisUtil::registerBufHandle(void *ptr,
                                         size_t size,
                                         int flags)
{
    hipFileError_t status;

    status = hipFileBufRegister(ptr, size, flags);
    if (status.err != hipFileSuccess) {
        NIXL_WARN << "AIS: buffer registration failed - will use compat mode";
    }
    return NIXL_SUCCESS;
}

nixl_status_t aisUtil::openAisDriver()
{
    hipFileError_t err;

    err = hipFileDriverOpen();
    if (err.err != hipFileSuccess) {
        NIXL_ERROR << "AIS: error initializing AMD Infinity Storage driver";
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

void aisUtil::closeAisDriver()
{
    (void)hipFileDriverClose();
}

void aisUtil::deregisterFileHandle(aisFileHandle& handle)
{
    (void)hipFileHandleDeregister(handle.hip_fhandle);
}

nixl_status_t aisUtil::deregisterBufHandle(void *ptr)
{
    hipFileError_t status;

    status = hipFileBufDeregister(ptr);
    if (status.err != hipFileSuccess) {
        NIXL_ERROR << "AIS: error de-registering buffer";
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

nixlAisIOBatch::nixlAisIOBatch(unsigned int size)
    : max_reqs(size)
{
    hipFileError_t err;

    io_batch_events = new hipFileIOEvents_t[size];
    io_batch_params = new hipFileIOParams_t[size];

    err = hipFileBatchIOSetUp(&batch_handle, size);
    if (err.err != 0) {
        NIXL_ERROR << "AIS: error in setting up batch";
        init_err = err;
    }
}

nixlAisIOBatch::~nixlAisIOBatch()
{
    if (current_status == NIXL_SUCCESS ||
        current_status == NIXL_ERR_NOT_POSTED) {
            delete[] io_batch_events;
            delete[] io_batch_params;
            (void)hipFileBatchIODestroy(batch_handle);
    } else {
            NIXL_ERROR << "AIS: attempting to delete a batch before completion";
    }
}

nixl_status_t nixlAisIOBatch::addToBatch(hipFileHandle_t fh, void *buffer,
                                         size_t size, size_t file_offset,
                                         size_t ptr_offset,
                                         hipFileOpcode_t type)
{
    hipFileIOParams_t *params = nullptr;

    if (batch_size >= max_reqs)
        return NIXL_ERR_BACKEND;

    params                          = &io_batch_params[batch_size];
    params->mode                    = hipFileBatch;
    params->fh                      = fh;
    params->u.batch.devPtr_base     = buffer;
    params->u.batch.file_offset     = file_offset;
    params->u.batch.devPtr_offset   = ptr_offset;
    params->u.batch.size            = size;
    params->opcode                  = type;
    params->cookie                  = params;
    batch_size++;

    return NIXL_SUCCESS;
}

nixl_status_t nixlAisIOBatch::cancelBatch()
{
    hipFileError_t err;

    err = hipFileBatchIOCancel(batch_handle);
    if (err.err != 0) {
        NIXL_ERROR << "AIS: error in canceling batch";
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

nixl_status_t nixlAisIOBatch::submitBatch(int flags)
{
    hipFileError_t err;

    err = hipFileBatchIOSubmit(batch_handle, batch_size,
                              io_batch_params, flags);
    if (err.err != 0) {
        NIXL_ERROR << "AIS: error in submitting batch";
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

nixl_status_t nixlAisIOBatch::checkStatus()
{
    hipFileError_t errBatch;
    unsigned int nr = batch_size;

    errBatch = hipFileBatchIOGetStatus(batch_handle, nr, &nr,
                                      io_batch_events, NULL);
    if (errBatch.err != 0) {
        NIXL_ERROR << "AIS: error in IO batch get status";
        current_status = NIXL_ERR_BACKEND;
    }

    entries_completed += nr;
    if (entries_completed < (unsigned int)batch_size)
        current_status = NIXL_IN_PROG;
    else if (entries_completed > batch_size)
        current_status = NIXL_ERR_UNKNOWN;
    else
        current_status = NIXL_SUCCESS;

    return current_status;
}

void nixlAisIOBatch::reset() {
    entries_completed = 0;
    batch_size = 0;
    current_status = NIXL_ERR_NOT_POSTED;
}
