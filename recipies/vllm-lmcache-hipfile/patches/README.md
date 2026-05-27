<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# LMCache patches (vllm-lmcache-hipfile image)

Patches are applied after `LMCACHE_SHA` checkout in the Dockerfile.

| File | Upstream | Purpose |
|------|----------|---------|
| `lmcache-pr-3008-cache-salt.patch` | [LMCache#3008][pr-3008] | Include vLLM `cache_salt` in `LMCacheConnectorV1` cache keys via `lmcache.tag.cachesalt` (fixes [#2878][issue-2878]). |
| `lmcache-storage-mode-switch.patch` | (rocm-aic) | Runtime `GET\|POST /storage/mode` to switch hipfile (`GdsBackend`) and posix (`RemoteBackend-fs`) without restarting vLLM. |
| `lmcache-gds-eviction-log.patch` | (rocm-aic) | Downgrade per-allocation GDS eviction notice from WARNING to DEBUG (hipfile has no eviction). |
| `lmcache-sha256-cbor-int.patch` | [LMCache#2979][issue-2979] | Convert vLLM `sha256_cbor` bytes hashes to int for LMCache keys (vLLM 0.19+). |
| `lmcache-controller-log.patch` | (rocm-aic) | Controller sender WARNING only when `enable_controller` is true. |
| `lmcache-chunk-statistics-hash.patch` | (rocm-aic) | Chunk statistics use `pre_caching_hash_algorithm` (same as KV keys). |

Adapted for the riley-dixon fork at `LMCACHE_SHA` (not a raw `git apply` of the upstream PR).

<!-- References -->

[pr-3008]: https://github.com/LMCache/LMCache/pull/3008
[issue-2878]: https://github.com/LMCache/LMCache/issues/2878
[issue-2979]: https://github.com/LMCache/LMCache/issues/2979
