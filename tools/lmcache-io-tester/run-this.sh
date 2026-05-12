#!/bin/bash

set -euo pipefail

python -m src.lmcache-sim run \
    --storage-type filesystem \
    --storage-path /opt/lmcache-test-fs \
    --pattern store-only \
    --num-operations 65536

sleep 3

python -m src.lmcache-sim run \
    --storage-type filesystem \
    --storage-path /opt/lmcache-test-fs \
    --pattern lookup-only \
    --fs-odirect \
    --duration 60
    
sleep 3

python -m src.lmcache-sim run \
    --storage-type filesystem \
    --storage-path /opt/lmcache-test-fs \
    --pattern retrieve-only \
    --fs-odirect \
    --duration 60

sleep 3

python -m src.lmcache-sim run \
    --storage-type local-disk \
    --storage-path /opt/lmcache-test-disk \
    --pattern store-only \
    --num-operations 16384

sleep 3
    
python -m src.lmcache-sim run \
    --storage-type local-disk \
    --storage-path /opt/lmcache-test-disk \
    --pattern retrieve-only \
    --duration 60