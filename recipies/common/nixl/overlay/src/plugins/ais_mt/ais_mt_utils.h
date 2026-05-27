/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef __AIS_MT_UTILS_H
#define __AIS_MT_UTILS_H

#include <fcntl.h>
#include <unistd.h>
#include <nixl.h>
#include <hipfile.h>

class aisMtFileHandle {
public:
    aisMtFileHandle (int fd);
    ~aisMtFileHandle();

    aisMtFileHandle (const aisMtFileHandle &) = delete;
    aisMtFileHandle &
    operator= (const aisMtFileHandle &) = delete;
    aisMtFileHandle (aisMtFileHandle &&) = delete;
    aisMtFileHandle &
    operator= (aisMtFileHandle &&) = delete;

    int fd{-1};
    hipFileHandle_t hip_fhandle{nullptr};
};

class aisMtMemBuf {
public:
    aisMtMemBuf (void *ptr, size_t sz, int flags = 0);
    ~aisMtMemBuf();

    aisMtMemBuf (const aisMtMemBuf &) = delete;
    aisMtMemBuf &
    operator= (const aisMtMemBuf &) = delete;
    aisMtMemBuf (aisMtMemBuf &&) = delete;
    aisMtMemBuf &
    operator= (aisMtMemBuf &&) = delete;

private:
    void *base_{nullptr};
    bool registered_{false};
};

class aisMtUtil {
public:
    aisMtUtil();
    ~aisMtUtil();
};
#endif
