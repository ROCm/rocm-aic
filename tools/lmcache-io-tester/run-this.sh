#!/bin/bash

set -euo pipefail

DATA_DIR="/mnt/rocm-icms-cache/stebates/lmcache-io-tester/data"

python -m src.lmcache-sim run \
    --storage-type filesystem \
    --storage-path $DATA_DIR \
    --pattern store-only \
    --num-operations 65536

sleep 3

python -m src.lmcache-sim run \
    --storage-type filesystem \
    --storage-path $DATA_DIR \
    --pattern lookup-only \
    --fs-odirect \
    --duration 60
    
sleep 3
exit 0

python -m src.lmcache-sim run \
    --storage-type filesystem \
    --storage-path $DATA_DIR \
    --pattern retrieve-only \
    --fs-odirect \
    --duration 60

sleep 3

python -m src.lmcache-sim run \
    --storage-type local-disk \
    --storage-path $DATA_DIR \
    --pattern store-only \
    --num-operations 16384

sleep 3
    
python -m src.lmcache-sim run \
    --storage-type local-disk \
    --storage-path $DATA_DIR \
    --pattern retrieve-only \
    --duration 60