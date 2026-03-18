#!/bin/bash

#
# Launch trace recipe with default libhipfile.so location
#

stdbuf -oL bpftrace -q scripts/hipfile.bt "${1:-/opt/rocm/lib/libhipfile.so.0}"
