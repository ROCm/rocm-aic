#! /usr/bin/env python3

from pathlib import Path
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import random
import sys
import hipfile
import time
import os


import ctypes
from ctypes import c_void_p, c_size_t, c_int, POINTER

class HIPError(Exception):
    """Exception raised for HIP errors"""
    pass

class HIP:
    """Python wrapper for HIP runtime API"""
    
    # HIP error codes
    hipSuccess = 0
    hipErrorMemoryAllocation = 2
    
    def __init__(self, lib_path=None):
        """Initialize HIP library"""
        if lib_path is None:
            # Try common library names
            for name in ["libamdhip64.so", "libamdhip64.so.5", "libamdhip64.so.6", "amdhip64.dll"]:
                try:
                    self.lib = ctypes.CDLL(name)
                    break
                except OSError:
                    continue
            else:
                raise OSError("Could not load HIP library")
        else:
            self.lib = ctypes.CDLL(lib_path)
        
        self._setup_functions()
    
    def _setup_functions(self):
        """Setup function signatures"""
        # hipMemAlloc
        self.lib.hipMalloc.argtypes = [POINTER(c_void_p), c_size_t]
        self.lib.hipMalloc.restype = c_int
        
        # hipMemFree
        self.lib.hipFree.argtypes = [c_void_p]
        self.lib.hipFree.restype = c_int
        
    def check_error(self, error_code, func_name="HIP function"):
        """Check error code and raise exception if not success"""
        if error_code != self.hipSuccess:
            raise HIPError(f"{func_name} failed: error code: {error_code}")
    
    def mem_alloc(self, size_bytes):
        """Allocate GPU memory"""
        dev_ptr = c_void_p()
        error = self.lib.hipMalloc(ctypes.byref(dev_ptr), c_size_t(size_bytes))
        self.check_error(error, "hipMemAlloc")
        return dev_ptr.value
    
    def mem_free(self, dev_ptr):
        """Free GPU memory"""
        if isinstance(dev_ptr, int):
            dev_ptr = c_void_p(dev_ptr)
        error = self.lib.hipFree(dev_ptr)
        self.check_error(error, "hipMemFree")


_hip_runtime = None


def _get_hip():
    """Single HIP ctypes wrapper for the process (avoid reloading in hot paths)."""
    global _hip_runtime
    if _hip_runtime is None:
        _hip_runtime = HIP()
    return _hip_runtime


class GpuBuffer:

    def __init__(self, nbytes):
        self._hip = _get_hip()
        self.addr = self._hip.mem_alloc(nbytes)
        self.nbytes = nbytes
        self._closed = False

    def close(self):
        if self._closed:
            return
        if self.addr is not None:
            self._hip.mem_free(self.addr)
            self.addr = None
        self._closed = True


def read(path, buffer_queue):
    nread = 0
    buffer = buffer_queue.get()
    nbytes = buffer._length

    t0_gbl = time.perf_counter()
    with hipfile.FileHandle(path, 0) as handle:
        t0_rd = time.perf_counter()
        nread = handle.read(buffer, nbytes, 4096, 0)
        t1_rd = time.perf_counter()
    t1_gbl = time.perf_counter()
    buffer_queue.put(buffer)
    gbl = t1_gbl - t0_gbl
    rd = t1_rd - t0_rd
    #print(f"{nread} {nbytes / gbl / 1024**2:.2f} {nbytes / rd / 1024**2:.2f}")
    return nread, gbl, rd 

if __name__ == "__main__":
    # Set these environment variables if they haven't already been set
    os.environ.setdefault('HIPFILE_UNSUPPORTED_FILE_SYSTEMS', 'true')
    os.environ.setdefault('HIPFILE_ALLOW_COMPAT_MODE', 'false')

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("CHUNK_DIR", type=Path, help="LM Cache Chunk Directory")
    parser.add_argument("NTHREADS", default=1, type=int, help="Number of IO threads")
    parser.add_argument("ITERATIONS", default=1, type=int, help="Number of test iterations")
    parser.add_argument("NCHUNKS", default=100, type=int, help="Number of chunks to load per iteration")
    args = parser.parse_args()

    print(f"chunk dir: {args.CHUNK_DIR}")
    print(f"nthreads: {args.NTHREADS}")

    kvchunks = list(args.CHUNK_DIR.rglob("*.safetensors"))
    print(f"found {len(kvchunks)} kvchunks")
    if 0 == len(kvchunks):
        print("No kvchunks found!")
        sys.exit(1)
    if args.NCHUNKS < 1:
        print("error: NCHUNKS must be at least 1", file=sys.stderr)
        sys.exit(1)
    if args.NCHUNKS > len(kvchunks):
        print(
            f"error: NCHUNKS={args.NCHUNKS} exceeds discovered "
            f"chunk file count ({len(kvchunks)})",
            file=sys.stderr,
        )
        sys.exit(1)

    # Each kvchunk file starts with a 4096 byte metadata header
    min_size = max_size = tot_size = kvchunks[0].stat().st_size - 4096
    for kvchunk in kvchunks[1:]:
        st_size = kvchunk.stat().st_size - 4096
        min_size = min(min_size, st_size)
        max_size = max(max_size, st_size)
        tot_size += st_size
    # asssume chunk_size is min_size
    chunk_size = min_size
    #print(f"chunk size: {chunk_size / 1024**2}MB")
    #print(f"min size: {min_size / 1024**2}MB")
    #print(f"max size: {max_size / 1024**2}MB")
    #print(f"avg size: {tot_size / len(kvchunks) / 1024**2}MB")
    #min_size = 9 * 1024**2

    buffer_queue = Queue()
    gpu_buffers = []
    try:
        for _ in range(args.NTHREADS + 16):
            gpu_buffers.append(GpuBuffer(chunk_size))
        for gpu_buffer in gpu_buffers:
            hfbuf = hipfile.Buffer(gpu_buffer.addr, gpu_buffer.nbytes, 0)
            hfbuf.register()
            buffer_queue.put(hfbuf)

        for _ in range(args.ITERATIONS):
            with ThreadPoolExecutor(max_workers=args.NTHREADS) as tp:
                t0 = time.perf_counter()
                results = \
                    tp.map(read, 
                        random.sample(kvchunks, args.NCHUNKS),
                        (buffer_queue for _ in range(args.NCHUNKS)))
                nbytes = 0
                time_rd = 0
                time_gbl = 0
                for nb, _t_gbl, t_rd in results:
                    nbytes += nb
                    time_rd += t_rd
                t1 = time.perf_counter()
                mbs = nbytes / 1024**2 / (t1 - t0)
                rd_mbs = nbytes / 1024**2 / time_rd
                print(f"Read {args.NCHUNKS} chunks, {nbytes / 1024**2}MiB in {t1 - t0:.4f} seconds. global: {mbs:.0f}MiB/s io: {rd_mbs:.0f}MiB/s")
            #time.sleep(1)

        print('done')
    finally:
        for b in gpu_buffers:
            b.close()
