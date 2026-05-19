# LMCache patches (vllm-radeon image)

Patches are applied after `LMCACHE_SHA` checkout in the Dockerfile.

| File | Upstream | Purpose |
|------|----------|---------|
| `lmcache-pr-3008-cache-salt.patch` | [LMCache#3008][pr-3008] | Include vLLM `cache_salt` in `LMCacheConnectorV1` cache keys via `lmcache.tag.cachesalt` (fixes [#2878][issue-2878]). |
| `lmcache-storage-mode-switch.patch` | (rocm-aic) | Runtime `GET\|POST /storage/mode` to switch hipfile (`GdsBackend`) and posix (`RemoteBackend-fs`) without restarting vLLM. |

Adapted for the riley-dixon fork at `LMCACHE_SHA` (not a raw `git apply` of the upstream PR).

[pr-3008]: https://github.com/LMCache/LMCache/pull/3008
[issue-2878]: https://github.com/LMCache/LMCache/issues/2878
