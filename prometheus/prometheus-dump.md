# AIC Prometheus Metrics Reference

**Generated:** 2026-09-22 09:28 UTC · **Total metrics:** 604 · **SHA:** `dd1afc7`

## Component Versions

| Component | Version |
|-----------|---------|
| version | `0.1.0` |
| rocm | `7.14.1` |
| aiter | `v0.1.22.post1` |
| flash-attention | `v2.8.3.post1` |
| lmcache | `v0.5.5` |
| nixl | `v1.4.1` |
| hipfile | `rocm-7.14.1` |
| hsa-snoop | `v1.1.1` |

## Running Containers

| Container | Image | Status |
|-----------|-------|--------|
| `aic-hsa-snoop` | `rocm-aic-ci-dd1afc7:0.1.0-rocm7.14.1-pytorch2.13-vllm0.29.0-aiter0.1.22.post1-fa2.8.3.post1-lmcache0.5.5-nixl1.4.1-hsasnoop1.1.1` | Up 16 seconds |
| `aic-amdgpu-exporter` | `rocm/device-metrics-exporter:v1.5.2` | Up About a minute |
| `aic-vllm-gpu0` | `rocm-aic-ci-dd1afc7:0.1.0-rocm7.14.1-pytorch2.13-vllm0.29.0-aiter0.1.22.post1-fa2.8.3.post1-lmcache0.5.5-nixl1.4.1-hsasnoop1.1.1` | Up About a minute |
| `aic-lmcache` | `rocm-aic-ci-dd1afc7:0.1.0-rocm7.14.1-pytorch2.13-vllm0.29.0-aiter0.1.22.post1-fa2.8.3.post1-lmcache0.5.5-nixl1.4.1-hsasnoop1.1.1` | Up 2 minutes (healthy) |
| `aic-client` | `python:3.12` | Up 2 minutes |
| `aic-prometheus` | `prom/prometheus:v3.14.0` | Up 2 minutes |
| `aic-grafana` | `grafana/grafana:13.2.2` | Up 2 minutes |
| `aic-lmcache-coordinator` | `rocm-aic-ci-dd1afc7:0.1.0-rocm7.14.1-pytorch2.13-vllm0.29.0-aiter0.1.22.post1-fa2.8.3.post1-lmcache0.5.5-nixl1.4.1-hsasnoop1.1.1` | Up 2 minutes (healthy) |
| `zealous_spence` | `rocm/atom-dev:ubuntu24.04_py3.12_pytorch_release_2.10.0_kimi_k3_agentic_0911` | Up 18 minutes |
| `priceless_benz` | `rocm/atom-dev:ubuntu24.04_py3.12_pytorch_release_2.10.0_kimi_k3_agentic_0911` | Up 18 minutes |
| `inspiring_chandrasekhar` | `rocm/atom-dev:ubuntu24.04_py3.12_pytorch_release_2.10.0_kimi_k3_agentic_0911` | Up 18 minutes |
| `devil_atom_sp_0922` | `rocm/atom-dev:latest` | Up 54 minutes |
| `buildx_buildkit_aic-cache0` | `moby/buildkit:buildx-stable-1` | Up 8 hours |

## Metrics Sources

- [vllm](#vllm) — 89 metrics
- [lmcache](#lmcache) — 25 metrics
- [nixl](#nixl) — 19 metrics
- [lmcache_coordinator](#lmcache-coordinator) — 7 metrics
- [hsa_snoop](#hsa-snoop) — 7 metrics
- [amdgpu_exporter](#amdgpu-exporter) — 132 metrics
- [prometheus](#prometheus) — 263 metrics
- [node_exporter](#node-exporter) — 62 metrics

## vllm

| Metric | Type | Description |
|--------|------|-------------|
| `http_request_duration_highr_seconds` | histogram | Latency with many buckets but no API specific labels. Made for more accurate percentile calculations.  |
| `http_request_duration_highr_seconds_created` | gauge | Latency with many buckets but no API specific labels. Made for more accurate percentile calculations.  |
| `http_request_duration_seconds` | histogram | Latency with only few buckets by handler. Made to be only used if aggregation by handler is important.  |
| `http_request_size_bytes` | summary | Content length of incoming requests by handler. Only value of header is respected. Otherwise ignored. No percentile calculated.  |
| `http_requests_total` | counter | Total number of requests by method, status and handler. |
| `http_response_size_bytes` | summary | Content length of outgoing responses by handler. Only value of header is respected. Otherwise ignored. No percentile calculated.  |
| `process_cpu_seconds_total` | counter | Total user and system CPU time spent in seconds. |
| `process_max_fds` | gauge | Maximum number of open file descriptors. |
| `process_open_fds` | gauge | Number of open file descriptors. |
| `process_resident_memory_bytes` | gauge | Resident memory size in bytes. |
| `process_start_time_seconds` | gauge | Start time of the process since unix epoch in seconds. |
| `process_virtual_memory_bytes` | gauge | Virtual memory size in bytes. |
| `python_gc_collections_total` | counter | Number of times this generation was collected |
| `python_gc_objects_collected_total` | counter | Objects collected during gc |
| `python_gc_objects_uncollectable_total` | counter | Uncollectable objects found during GC |
| `python_info` | gauge | Python platform information |
| `vllm:cache_config_info` | gauge | Information of the LLMEngine CacheConfig |
| `vllm:e2e_request_latency_seconds` | histogram | Histogram of e2e request latency in seconds. |
| `vllm:e2e_request_latency_seconds_created` | gauge | Histogram of e2e request latency in seconds. |
| `vllm:engine_sleep_state` | gauge | Engine sleep state; awake = 0 means engine is sleeping; awake = 1 means engine is awake; weights_offloaded = 1 means sleep level 1; discard_all = 1 means sleep level 2. |
| `vllm:estimated_flops_per_gpu_created` | gauge | Estimated number of floating point operations per GPU (for Model Flops Utilization calculations). |
| `vllm:estimated_flops_per_gpu_total` | counter | Estimated number of floating point operations per GPU (for Model Flops Utilization calculations). |
| `vllm:estimated_read_bytes_per_gpu_created` | gauge | Estimated number of bytes read from memory per GPU (for Model Flops Utilization calculations). |
| `vllm:estimated_read_bytes_per_gpu_total` | counter | Estimated number of bytes read from memory per GPU (for Model Flops Utilization calculations). |
| `vllm:estimated_write_bytes_per_gpu_created` | gauge | Estimated number of bytes written to memory per GPU (for Model Flops Utilization calculations). |
| `vllm:estimated_write_bytes_per_gpu_total` | counter | Estimated number of bytes written to memory per GPU (for Model Flops Utilization calculations). |
| `vllm:external_prefix_cache_hits_created` | gauge | External prefix cache hits from KV connector cross-instance cache sharing, in terms of number of cached tokens. |
| `vllm:external_prefix_cache_hits_total` | counter | External prefix cache hits from KV connector cross-instance cache sharing, in terms of number of cached tokens. |
| `vllm:external_prefix_cache_queries_created` | gauge | External prefix cache queries from KV connector cross-instance cache sharing, in terms of number of queried tokens. |
| `vllm:external_prefix_cache_queries_total` | counter | External prefix cache queries from KV connector cross-instance cache sharing, in terms of number of queried tokens. |
| `vllm:generation_tokens_created` | gauge | Number of generation tokens processed. |
| `vllm:generation_tokens_total` | counter | Number of generation tokens processed. |
| `vllm:inter_token_latency_seconds` | histogram | Histogram of inter-token latency in seconds. |
| `vllm:inter_token_latency_seconds_created` | gauge | Histogram of inter-token latency in seconds. |
| `vllm:iteration_tokens_total` | histogram | Histogram of number of tokens per engine_step. |
| `vllm:iteration_tokens_total_created` | gauge | Histogram of number of tokens per engine_step. |
| `vllm:kv_block_idle_before_evict_seconds` | histogram | Histogram of idle time before KV cache block eviction. Sampled metrics (controlled by --kv-cache-metrics-sample). |
| `vllm:kv_block_idle_before_evict_seconds_created` | gauge | Histogram of idle time before KV cache block eviction. Sampled metrics (controlled by --kv-cache-metrics-sample). |
| `vllm:kv_block_lifetime_seconds` | histogram | Histogram of KV cache block lifetime from allocation to eviction. Sampled metrics (controlled by --kv-cache-metrics-sample). |
| `vllm:kv_block_lifetime_seconds_created` | gauge | Histogram of KV cache block lifetime from allocation to eviction. Sampled metrics (controlled by --kv-cache-metrics-sample). |
| `vllm:kv_block_reuse_gap_seconds` | histogram | Histogram of time gaps between consecutive KV cache block accesses. Only the most recent accesses are recorded (ring buffer). Sampled metrics (controlled by --kv-cache-metrics-sample). |
| `vllm:kv_block_reuse_gap_seconds_created` | gauge | Histogram of time gaps between consecutive KV cache block accesses. Only the most recent accesses are recorded (ring buffer). Sampled metrics (controlled by --kv-cache-metrics-sample). |
| `vllm:kv_cache_usage_perc` | gauge | KV-cache usage. 1 means 100 percent usage. |
| `vllm:mm_cache_hits_created` | gauge | Multi-modal cache hits, in terms of number of cached items. |
| `vllm:mm_cache_hits_total` | counter | Multi-modal cache hits, in terms of number of cached items. |
| `vllm:mm_cache_queries_created` | gauge | Multi-modal cache queries, in terms of number of queried items. |
| `vllm:mm_cache_queries_total` | counter | Multi-modal cache queries, in terms of number of queried items. |
| `vllm:model_weights_memory_bytes` | gauge | GPU memory used by model weights in bytes (static after load). |
| `vllm:num_preemptions_created` | gauge | Cumulative number of preemption from the engine. |
| `vllm:num_preemptions_total` | counter | Cumulative number of preemption from the engine. |
| `vllm:num_requests_running` | gauge | Number of requests in model execution batches. |
| `vllm:num_requests_waiting` | gauge | Number of requests waiting to be processed. |
| `vllm:num_requests_waiting_by_reason` | gauge | Number of waiting requests by reason. Reason labels: 'capacity' = waiting for scheduling capacity; 'deferred' = deferred by transient constraints (LoRA budget, KV transfer, blocked status). Sum of all reasons equals vllm:num_requests_waiting. |
| `vllm:prefix_cache_hits_created` | gauge | Prefix cache hits, in terms of number of cached tokens. |
| `vllm:prefix_cache_hits_total` | counter | Prefix cache hits, in terms of number of cached tokens. |
| `vllm:prefix_cache_queries_created` | gauge | Prefix cache queries, in terms of number of queried tokens. |
| `vllm:prefix_cache_queries_total` | counter | Prefix cache queries, in terms of number of queried tokens. |
| `vllm:prompt_tokens_by_source_created` | gauge | Number of prompt tokens by source. |
| `vllm:prompt_tokens_by_source_total` | counter | Number of prompt tokens by source. |
| `vllm:prompt_tokens_cached_created` | gauge | Number of cached prompt tokens (local + external). |
| `vllm:prompt_tokens_cached_total` | counter | Number of cached prompt tokens (local + external). |
| `vllm:prompt_tokens_created` | gauge | Number of prefill tokens processed. |
| `vllm:prompt_tokens_total` | counter | Number of prefill tokens processed. |
| `vllm:request_decode_time_seconds` | histogram | Histogram of time spent in DECODE phase for request. |
| `vllm:request_decode_time_seconds_created` | gauge | Histogram of time spent in DECODE phase for request. |
| `vllm:request_generation_tokens` | histogram | Number of generation tokens processed. |
| `vllm:request_generation_tokens_created` | gauge | Number of generation tokens processed. |
| `vllm:request_inference_time_seconds` | histogram | Histogram of time spent in RUNNING phase for request. |
| `vllm:request_inference_time_seconds_created` | gauge | Histogram of time spent in RUNNING phase for request. |
| `vllm:request_max_num_generation_tokens` | histogram | Histogram of maximum number of requested generation tokens. |
| `vllm:request_max_num_generation_tokens_created` | gauge | Histogram of maximum number of requested generation tokens. |
| `vllm:request_params_max_tokens` | histogram | Histogram of the max_tokens request parameter. |
| `vllm:request_params_max_tokens_created` | gauge | Histogram of the max_tokens request parameter. |
| `vllm:request_params_n` | histogram | Histogram of the n request parameter. |
| `vllm:request_params_n_created` | gauge | Histogram of the n request parameter. |
| `vllm:request_prefill_kv_computed_tokens` | histogram | Histogram of new KV tokens computed during prefill (excluding cached tokens). |
| `vllm:request_prefill_kv_computed_tokens_created` | gauge | Histogram of new KV tokens computed during prefill (excluding cached tokens). |
| `vllm:request_prefill_time_seconds` | histogram | Histogram of time spent in PREFILL phase for request. |
| `vllm:request_prefill_time_seconds_created` | gauge | Histogram of time spent in PREFILL phase for request. |
| `vllm:request_prompt_tokens` | histogram | Number of prefill tokens processed. |
| `vllm:request_prompt_tokens_created` | gauge | Number of prefill tokens processed. |
| `vllm:request_queue_time_seconds` | histogram | Histogram of time spent in WAITING phase for request. |
| `vllm:request_queue_time_seconds_created` | gauge | Histogram of time spent in WAITING phase for request. |
| `vllm:request_success_created` | gauge | Count of successfully processed requests. |
| `vllm:request_success_total` | counter | Count of successfully processed requests. |
| `vllm:request_time_per_output_token_seconds` | histogram | Histogram of time_per_output_token_seconds per request. |
| `vllm:request_time_per_output_token_seconds_created` | gauge | Histogram of time_per_output_token_seconds per request. |
| `vllm:time_to_first_token_seconds` | histogram | Histogram of time to first token in seconds. |
| `vllm:time_to_first_token_seconds_created` | gauge | Histogram of time to first token in seconds. |

## lmcache

| Metric | Type | Description |
|--------|------|-------------|
| `lmcache_mp_active_p2p_lookup_jobs` | gauge | Number of active P2P lookup jobs |
| `lmcache_mp_active_prefetch_jobs` | gauge | Number of active prefetch jobs |
| `lmcache_mp_event_bus_drain_lag_seconds` | gauge | Seconds since the oldest queued event was published; 0.0 when empty.  Rising values mean the drain thread is falling behind. |
| `lmcache_mp_event_bus_dropped_events_total` | counter | Cumulative events dropped because the EventBus queue was at max_queue_size. |
| `lmcache_mp_event_bus_queue_depth` | gauge | Events currently queued in the EventBus. |
| `lmcache_mp_l1_eviction_loop_ticks_total` | counter | L1 eviction-loop iterations (every cycle) |
| `lmcache_mp_l1_memory_usage_bytes` | gauge | Bytes currently held in L1 cache |
| `lmcache_mp_l1_usage_ratio` | gauge | L1 used/total ratio (0.0-1.0) |
| `lmcache_mp_l2_prefetch_adapters` | gauge | Count of L2 adapters attached to the prefetch controller, tagged by ``state`` (active or draining). |
| `lmcache_mp_l2_store_adapters` | gauge | Count of L2 adapters attached to the store controller, tagged by ``state`` (active or draining). |
| `lmcache_mp_l2_usage_bytes` | gauge | Bytes currently held in each L2 adapter, tagged by ``l2_name`` (one observation per adapter). |
| `lmcache_mp_prefetch_hit_chunks_total` | counter | Chunks served from cache by tier (l1=DRAM, l2=NVMe). |
| `lmcache_mp_prefetch_miss_chunks_total` | counter | Chunks requested but not found in any cache tier. |
| `lmcache_mp_prefetch_requests_total` | counter | Total completed prefetch lookup requests. |
| `process_cpu_seconds_total` | counter | Total user and system CPU time spent in seconds. |
| `process_max_fds` | gauge | Maximum number of open file descriptors. |
| `process_open_fds` | gauge | Number of open file descriptors. |
| `process_resident_memory_bytes` | gauge | Resident memory size in bytes. |
| `process_start_time_seconds` | gauge | Start time of the process since unix epoch in seconds. |
| `process_virtual_memory_bytes` | gauge | Virtual memory size in bytes. |
| `python_gc_collections_total` | counter | Number of times this generation was collected |
| `python_gc_objects_collected_total` | counter | Objects collected during gc |
| `python_gc_objects_uncollectable_total` | counter | Uncollectable objects found during GC |
| `python_info` | gauge | Python platform information |
| `target_info` | gauge | Target metadata |

## nixl

| Metric | Type | Description |
|--------|------|-------------|
| `agent_errors_total` | counter | Cumulative error count by status |
| `agent_memory_deregistered_last_bytes` | gauge | Memory deregistered by the last operation |
| `agent_memory_deregistered_total` | counter | Cumulative memory deregistered |
| `agent_memory_registered_last_bytes` | gauge | Memory registered by the last operation |
| `agent_memory_registered_total` | counter | Cumulative memory registered |
| `agent_rx_bytes_total` | counter | Number of bytes received by the agent |
| `agent_rx_last_bytes` | gauge | Bytes received by the last request |
| `agent_rx_requests_num_total` | counter | Number of requests received by the agent |
| `agent_telemetry_events_dropped_total` | counter | Cumulative telemetry events dropped at the producer-side staging queue |
| `agent_tx_bytes_total` | counter | Number of bytes sent by the agent |
| `agent_tx_last_bytes` | gauge | Bytes sent by the last request |
| `agent_tx_requests_num_total` | counter | Number of requests sent by the agent |
| `agent_xfer_post_time` | gauge | Post time of the last request |
| `agent_xfer_post_time_total` | counter | Cumulative sum of time from start to posting to the back-end |
| `agent_xfer_time` | gauge | Transfer time of the last request |
| `agent_xfer_time_total` | counter | Cumulative sum of transfer time from start to completion |
| `exposer_request_latencies` | summary | Latencies of serving scrape requests, in microseconds |
| `exposer_scrapes_total` | counter | Number of times metrics were scraped |
| `exposer_transferred_bytes_total` | counter | Transferred bytes to metrics services |

## lmcache_coordinator

| Metric | Type | Description |
|--------|------|-------------|
| `lmcache_coordinator_directory_keys_total` | gauge | Keys with at least one placement in the key directory |
| `lmcache_coordinator_directory_placements_total` | gauge | Total placements across all tracked keys |
| `lmcache_coordinator_instance_info` | gauge | Non-zero for each registered instance; labels carry metadata |
| `lmcache_coordinator_instances_registered` | gauge | Number of MP servers currently registered with the coordinator |
| `lmcache_coordinator_quota_usage_bytes_total` | gauge | Total L2 usage bytes tracked by coordinator |
| `lmcache_coordinator_scrape_duration_seconds` | gauge | Time in seconds to scrape the LMCache coordinator REST API |
| `lmcache_coordinator_up` | gauge | 1 if the LMCache MP coordinator /healthz reports healthy |

## hsa_snoop

| Metric | Type | Description |
|--------|------|-------------|
| `ais_active` | gauge | 1 if at least one AIS IO operation has been observed since startup, 0 otherwise; latches at 1 |
| `exposer_request_latencies` | summary | Latencies of serving scrape requests, in microseconds |
| `exposer_scrapes_total` | counter | Number of times metrics were scraped |
| `exposer_transferred_bytes_total` | counter | Transferred bytes to metrics services |
| `hsa_active_queues` | gauge | Number of currently tracked AQL queues |
| `hsa_sdma_present` | gauge | 1 if at least one SDMA queue has been observed on this system, 0 otherwise; always 0 on APU nodes (e.g. Strix Halo) where no discrete SDMA engine exists |
| `hsa_snoop_up` | gauge | 1 if hsa-snoop is running and attached, 0 otherwise |

## amdgpu_exporter

| Metric | Type | Description |
|--------|------|-------------|
| `gpu_afid_errors` | gauge | Last Occurred RAS Event associated AMD Field Identifier list |
| `gpu_clock` | gauge | List of current GPU clock frequencies in MHz |
| `gpu_ecc_correct_athub` | gauge | Correctable error count in ATHUB block |
| `gpu_ecc_correct_bif` | gauge | Correctable error count in BIF block |
| `gpu_ecc_correct_df` | gauge | Correctable error count in DF block |
| `gpu_ecc_correct_fuse` | gauge | Correctable error count in Fuse block |
| `gpu_ecc_correct_gfx` | gauge | Correctable error count in GFX block |
| `gpu_ecc_correct_hdp` | gauge | Correctable error count in HDP block |
| `gpu_ecc_correct_ih` | gauge | Correctable error count in IH block |
| `gpu_ecc_correct_jpeg` | gauge | Correctable error count in JPEG block |
| `gpu_ecc_correct_mca` | gauge | Correctable error count in MCA block |
| `gpu_ecc_correct_mmhub` | gauge | Correctable error count in MMHUB block |
| `gpu_ecc_correct_mp0` | gauge | Correctable error count in MP0 block |
| `gpu_ecc_correct_mp1` | gauge | Correctable error count in MP1 block |
| `gpu_ecc_correct_mpio` | gauge | Correctable error count in MPIO block |
| `gpu_ecc_correct_sdma` | gauge | Correctable error count in SDMA block |
| `gpu_ecc_correct_sem` | gauge | Correctable error count in SEM block |
| `gpu_ecc_correct_smn` | gauge | Correctable error count in SMN block |
| `gpu_ecc_correct_total` | gauge | Total Correctable error count |
| `gpu_ecc_correct_umc` | gauge | Correctable error count in UMC block |
| `gpu_ecc_correct_vcn` | gauge | Correctable error count in VCN block |
| `gpu_ecc_correct_xgmi_wafl` | gauge | Correctable error count in WAFL block |
| `gpu_ecc_deferred_athub` | gauge | Accumulated deferred ECC errors in ATHUB block |
| `gpu_ecc_deferred_bif` | gauge | Accumulated deferred ECC errors in BIF block |
| `gpu_ecc_deferred_df` | gauge | Accumulated deferred ECC errors in DF block |
| `gpu_ecc_deferred_fuse` | gauge | Accumulated deferred ECC errors in FUSE block |
| `gpu_ecc_deferred_gfx` | gauge | Accumulated deferred ECC errors in GFX block |
| `gpu_ecc_deferred_hdp` | gauge | Accumulated deferred ECC errors in HDP block |
| `gpu_ecc_deferred_ih` | gauge | Accumulated deferred ECC errors in IH block |
| `gpu_ecc_deferred_jpeg` | gauge | Accumulated deferred ECC errors in JPEG block |
| `gpu_ecc_deferred_mca` | gauge | Accumulated deferred ECC errors in MCA block |
| `gpu_ecc_deferred_mmhub` | gauge | Accumulated deferred ECC errors in MMHUB block |
| `gpu_ecc_deferred_mp0` | gauge | Accumulated deferred ECC errors in MP0 block |
| `gpu_ecc_deferred_mp1` | gauge | Accumulated deferred ECC errors in MP1 block |
| `gpu_ecc_deferred_mpio` | gauge | Accumulated deferred ECC errors in MPIO block |
| `gpu_ecc_deferred_sdma` | gauge | Accumulated deferred ECC errors in SDMA block |
| `gpu_ecc_deferred_sem` | gauge | Accumulated deferred ECC errors in SEM block |
| `gpu_ecc_deferred_smn` | gauge | Accumulated deferred ECC errors in SMN block |
| `gpu_ecc_deferred_total` | gauge | Total accumulated deferred ECC errors across all GPU blocks |
| `gpu_ecc_deferred_umc` | gauge | Accumulated deferred ECC errors in UMC block |
| `gpu_ecc_deferred_vcn` | gauge | Accumulated deferred ECC errors in VCN block |
| `gpu_ecc_deferred_xgmi_wafl` | gauge | Accumulated deferred ECC errors in XGMI WAFL block |
| `gpu_ecc_uncorrect_athub` | gauge | Uncorrectable error count in ATHUB block |
| `gpu_ecc_uncorrect_bif` | gauge | Uncorrectable error count in BIF block |
| `gpu_ecc_uncorrect_df` | gauge | Uncorrectable error count in DF block |
| `gpu_ecc_uncorrect_fuse` | gauge | Uncorrectable error count in Fuse block |
| `gpu_ecc_uncorrect_gfx` | gauge | Uncorrectable error count in GFX block |
| `gpu_ecc_uncorrect_hdp` | gauge | Uncorrectable error count in HDP block |
| `gpu_ecc_uncorrect_ih` | gauge | Uncorrectable error count in IH block |
| `gpu_ecc_uncorrect_jpeg` | gauge | Uncorrectable error count in JPEG block |
| `gpu_ecc_uncorrect_mca` | gauge | Uncorrectable error count in MCA block |
| `gpu_ecc_uncorrect_mmhub` | gauge | Uncorrectable error count in MMHUB block |
| `gpu_ecc_uncorrect_mp0` | gauge | Uncorrectable error count in MP0 block |
| `gpu_ecc_uncorrect_mp1` | gauge | Uncorrectable error count in MP1 block |
| `gpu_ecc_uncorrect_mpio` | gauge | Uncorrectable error count in MPIO block |
| `gpu_ecc_uncorrect_sdma` | gauge | Uncorrectable error count in SDMA block |
| `gpu_ecc_uncorrect_sem` | gauge | Uncorrectable error count in SEM block |
| `gpu_ecc_uncorrect_smn` | gauge | Uncorrectable error count in SMN block |
| `gpu_ecc_uncorrect_total` | gauge | Total Uncorrectable error count |
| `gpu_ecc_uncorrect_umc` | gauge | Uncorrectable error count in UMC block |
| `gpu_ecc_uncorrect_vcn` | gauge | Uncorrectable error count in VCN block |
| `gpu_ecc_uncorrect_xgmi_wafl` | gauge | Uncorrectable error count in WAFL block |
| `gpu_energy_consumed` | gauge | Accumulated energy consumed by the GPU in uJ |
| `gpu_free_gtt` | gauge | Free graphics translation table memory of the GPU (in MB) |
| `gpu_free_visible_vram` | gauge | Free visible VRAM memory of the GPU (in MB) |
| `gpu_free_vram` | gauge | Free VRAM memory of the GPU (in MB) |
| `gpu_gfx_activity` | gauge | Graphics engine usage in Percentage (0-100) |
| `gpu_gfx_busy_instantaneous` | gauge | Gfx busy instantaneous per accelerated compute processor(xcp) per compute core (xcc), as per partitioning of the system |
| `gpu_health` | gauge | Health of the GPU (0 = Unhealthy \| 1 = Healthy) |
| `gpu_jpeg_busy_instantaneous` | gauge | Jpeg busy instantaneous per accelerated compute processor(xcp) per compute core (xcc), as per partitioning of the system |
| `gpu_junction_temperature` | gauge | Current junction/hotspot temperature in Celsius |
| `gpu_max_clock` | gauge | List of current GPU max clock frequencies in MHz |
| `gpu_memory_temperature` | gauge | Current memory temperature in Celsius |
| `gpu_min_clock` | gauge | List of current GPU min clock frequencies in MHz |
| `gpu_nodes_total` | gauge | Number of GPUs in the node |
| `gpu_package_power` | gauge | Current socket power in Watts |
| `gpu_power_usage` | gauge | GPU Power usage in Watts |
| `gpu_process_cu_occupancy` | gauge | Compute Unit occupancy for a process in percent |
| `gpu_total_gtt` | gauge | Total graphics translation table memory of the GPU (in MB) |
| `gpu_total_visible_vram` | gauge | Total visible VRAM memory of the GPU (in MB) |
| `gpu_total_vram` | gauge | Total VRAM memory of the GPU (in MB) |
| `gpu_umc_activity` | gauge | Memory engine usage in Percentage (0-100) |
| `gpu_used_gtt` | gauge | Used graphics translation table memory of the GPU (in MB) |
| `gpu_used_visible_vram` | gauge | Used visible VRAM memory of the GPU (in MB) |
| `gpu_used_vram` | gauge | Used VRAM memory of the GPU (in MB) |
| `gpu_vcn_busy_instantaneous` | gauge | Vcn busy instantaneous per accelerated compute processor(xcp) per compute core (xcc), as per partitioning of the system |
| `gpu_violation_current_accumulated_counter` | gauge | current accumulated violation counter |
| `gpu_violation_gfx_clock_below_host_limit_power_accumulated` | gauge | GFX clock below host limit power accumulated violation counter |
| `gpu_violation_gfx_clock_below_host_limit_power_percentage` | gauge | GFX clock below host limit power percentage violation counter |
| `gpu_violation_gfx_clock_below_host_limit_thermal_accumulated` | gauge | GFX clock below host limit thermal accumulated violation counter |
| `gpu_violation_gfx_clock_below_host_limit_thermal_percentage` | gauge | GFX clock below host limit thermal percentage violation counter |
| `gpu_violation_gfx_clock_below_host_limit_total_accumulated` | gauge | GFX clock below host limit total accumulated violation counter |
| `gpu_violation_gfx_clock_below_host_limit_total_percentage` | gauge | GFX clock below host limit total percentage violation counter |
| `gpu_violation_hbm_thermal_residency_accumulated` | gauge | HBM accumulated violation counter |
| `gpu_violation_hbm_thermal_residency_percentage` | gauge | HBM percentage violation counter |
| `gpu_violation_low_utilization_accumulated` | gauge | GPU low utilization accumulated violation counter |
| `gpu_violation_low_utilization_percentage` | gauge | GPU utilization percentage violation counter |
| `gpu_violation_ppt_residency_accumulated` | gauge | package power tracking accumulated violation counter |
| `gpu_violation_ppt_residency_percentage` | gauge | package power tracking percentage violation counter |
| `gpu_violation_processor_hot_residency_accumulated` | gauge | process hot residency accumulated violation counter |
| `gpu_violation_processor_hot_residency_percentage` | gauge | process hot residency percentage violation counter |
| `gpu_violation_socket_thermal_residency_accumulated` | gauge | socket thermal accumulated violation counter |
| `gpu_violation_socket_thermal_residency_percentage` | gauge | socket thermal percentage violation counter |
| `gpu_violation_vr_thermal_residency_accumulated` | gauge | voltage rail accumulated violation counter |
| `gpu_violation_vr_thermal_residency_percentage` | gauge | voltage rail percentage violation counter |
| `gpu_vram_max_bandwidth` | gauge | GPU VRAM maximum bandwidth at max memory clock in GB/s |
| `gpu_xgmi_link_rx` | gauge | Accumulated XGMI Link Data Read in KB |
| `gpu_xgmi_link_tx` | gauge | Accumulated XGMI Link Data Write in KB |
| `gpu_xgmi_nbr_0_beats_tx` | gauge | Data beats sent to neighbor 0; Each beat represents 32 bytes |
| `gpu_xgmi_nbr_0_nop_tx` | gauge | NOPs sent to neighbor 0 |
| `gpu_xgmi_nbr_0_req_tx` | gauge | Outgoing requests to neighbor 0 |
| `gpu_xgmi_nbr_0_resp_tx` | gauge | Outgoing responses to neighbor 0 |
| `gpu_xgmi_nbr_0_tx_thrput` | gauge | Represents the number of outbound beats (each representing 32 bytes) on link 0; Throughput = BEATS/time_running * 10^9  bytes/sec |
| `gpu_xgmi_nbr_1_beats_tx` | gauge | Data beats sent to neighbor 1; Each beat represents 32 bytes |
| `gpu_xgmi_nbr_1_nop_tx` | gauge | NOPs sent to neighbor 1 |
| `gpu_xgmi_nbr_1_req_tx` | gauge | Outgoing requests to neighbor 1 |
| `gpu_xgmi_nbr_1_resp_tx` | gauge | Outgoing responses to neighbor 1 |
| `gpu_xgmi_nbr_1_tx_thrput` | gauge | Represents the number of outbound beats (each representing 32 bytes) on link 1; Throughput = BEATS/time_running * 10^9  bytes/sec |
| `gpu_xgmi_nbr_2_tx_thrput` | gauge | Represents the number of outbound beats (each representing 32 bytes) on link 2; Throughput = BEATS/time_running * 10^9  bytes/sec |
| `gpu_xgmi_nbr_3_tx_thrput` | gauge | Represents the number of outbound beats (each representing 32 bytes) on link 3; Throughput = BEATS/time_running * 10^9  bytes/sec |
| `gpu_xgmi_nbr_4_tx_thrput` | gauge | Represents the number of outbound beats (each representing 32 bytes) on link 4; Throughput = BEATS/time_running * 10^9  bytes/sec |
| `gpu_xgmi_nbr_5_tx_thrput` | gauge | Represents the number of outbound beats (each representing 32 bytes) on link 5; Throughput = BEATS/time_running * 10^9  bytes/sec |
| `pcie_bandwidth` | gauge | Current PCIe bandwidth in Mb/s |
| `pcie_bidirectional_bandwidth` | gauge | Accumulated bandwidth on PCIe link in GB/sec |
| `pcie_max_speed` | gauge | Maximum PCIe speed in GT/s |
| `pcie_nack_received_count` | gauge | PCIe NAK received accumulated count |
| `pcie_nack_sent_count` | gauge | PCIe NAK sent accumulated count |
| `pcie_recovery_count` | gauge | Total number of PCIe recoveries |
| `pcie_replay_count` | gauge | Total number of PCIe replays |
| `pcie_replay_rollover_count` | gauge | PCIe replay accumulated count |
| `pcie_speed` | gauge | Current PCIe speed in GT/s |
| `promhttp_metric_handler_errors_total` | counter | Total number of internal errors encountered by the promhttp metric handler. |

## prometheus

| Metric | Type | Description |
|--------|------|-------------|
| `go_gc_cleanups_executed_cleanups_total` | counter | Approximate total count of cleanup functions (created by runtime.AddCleanup) executed by the runtime. Subtract /gc/cleanups/queued:cleanups to approximate cleanup queue length. Useful for detecting slow cleanups holding up the queue. Sourced from /gc/cleanups/executed:cleanups. |
| `go_gc_cleanups_queued_cleanups_total` | counter | Approximate total count of cleanup functions (created by runtime.AddCleanup) queued by the runtime for execution. Subtract from /gc/cleanups/executed:cleanups to approximate cleanup queue length. Useful for detecting slow cleanups holding up the queue. Sourced from /gc/cleanups/queued:cleanups. |
| `go_gc_cycles_automatic_gc_cycles_total` | counter | Count of completed GC cycles generated by the Go runtime. Sourced from /gc/cycles/automatic:gc-cycles. |
| `go_gc_cycles_forced_gc_cycles_total` | counter | Count of completed GC cycles forced by the application. Sourced from /gc/cycles/forced:gc-cycles. |
| `go_gc_cycles_total_gc_cycles_total` | counter | Count of all completed GC cycles. Sourced from /gc/cycles/total:gc-cycles. |
| `go_gc_duration_seconds` | summary | A summary of the wall-time pause (stop-the-world) duration in garbage collection cycles. |
| `go_gc_finalizers_executed_finalizers_total` | counter | Total count of finalizer functions (created by runtime.SetFinalizer) executed by the runtime. Subtract /gc/finalizers/queued:finalizers to approximate finalizer queue length. Useful for detecting finalizers overwhelming the queue, either by being too slow, or by there being too many of them. Sourced from /gc/finalizers/executed:finalizers. |
| `go_gc_finalizers_queued_finalizers_total` | counter | Total count of finalizer functions (created by runtime.SetFinalizer) and queued by the runtime for execution. Subtract from /gc/finalizers/executed:finalizers to approximate finalizer queue length. Useful for detecting slow finalizers holding up the queue. Sourced from /gc/finalizers/queued:finalizers. |
| `go_gc_gogc_percent` | gauge | Heap size target percentage configured by the user, otherwise 100. This value is set by the GOGC environment variable, and the runtime/debug.SetGCPercent function. Sourced from /gc/gogc:percent. |
| `go_gc_gomemlimit_bytes` | gauge | Go runtime memory limit configured by the user, otherwise math.MaxInt64. This value is set by the GOMEMLIMIT environment variable, and the runtime/debug.SetMemoryLimit function. Sourced from /gc/gomemlimit:bytes. |
| `go_gc_heap_allocs_by_size_bytes` | histogram | Distribution of heap allocations by approximate size. Bucket counts increase monotonically. Note that this does not include tiny objects as defined by /gc/heap/tiny/allocs:objects, only tiny blocks. Sourced from /gc/heap/allocs-by-size:bytes. |
| `go_gc_heap_allocs_bytes_total` | counter | Cumulative sum of memory allocated to the heap by the application. Sourced from /gc/heap/allocs:bytes. |
| `go_gc_heap_allocs_objects_total` | counter | Cumulative count of heap allocations triggered by the application. Note that this does not include tiny objects as defined by /gc/heap/tiny/allocs:objects, only tiny blocks. Sourced from /gc/heap/allocs:objects. |
| `go_gc_heap_frees_by_size_bytes` | histogram | Distribution of freed heap allocations by approximate size. Bucket counts increase monotonically. Note that this does not include tiny objects as defined by /gc/heap/tiny/allocs:objects, only tiny blocks. Sourced from /gc/heap/frees-by-size:bytes. |
| `go_gc_heap_frees_bytes_total` | counter | Cumulative sum of heap memory freed by the garbage collector. Sourced from /gc/heap/frees:bytes. |
| `go_gc_heap_frees_objects_total` | counter | Cumulative count of heap allocations whose storage was freed by the garbage collector. Note that this does not include tiny objects as defined by /gc/heap/tiny/allocs:objects, only tiny blocks. Sourced from /gc/heap/frees:objects. |
| `go_gc_heap_goal_bytes` | gauge | Heap size target for the end of the GC cycle. Sourced from /gc/heap/goal:bytes. |
| `go_gc_heap_live_bytes` | gauge | Heap memory occupied by live objects that were marked by the previous GC. Sourced from /gc/heap/live:bytes. |
| `go_gc_heap_objects_objects` | gauge | Number of objects, live or unswept, occupying heap memory. Sourced from /gc/heap/objects:objects. |
| `go_gc_heap_tiny_allocs_objects_total` | counter | Count of small allocations that are packed together into blocks. These allocations are counted separately from other allocations because each individual allocation is not tracked by the runtime, only their block. Each block is already accounted for in allocs-by-size and frees-by-size. Sourced from /gc/heap/tiny/allocs:objects. |
| `go_gc_limiter_last_enabled_gc_cycle` | gauge | GC cycle the last time the GC CPU limiter was enabled. This metric is useful for diagnosing the root cause of an out-of-memory error, because the limiter trades memory for CPU time when the GC's CPU time gets too high. This is most likely to occur with use of SetMemoryLimit. The first GC cycle is cycle 1, so a value of 0 indicates that it was never enabled. Sourced from /gc/limiter/last-enabled:gc-cycle. |
| `go_gc_pauses_seconds` | histogram | Deprecated. Prefer the identical /sched/pauses/total/gc:seconds. Sourced from /gc/pauses:seconds. |
| `go_gc_scan_globals_bytes` | gauge | The total amount of global variable space that is scannable. Sourced from /gc/scan/globals:bytes. |
| `go_gc_scan_heap_bytes` | gauge | The total amount of heap space that is scannable. Sourced from /gc/scan/heap:bytes. |
| `go_gc_scan_stack_bytes` | gauge | The number of bytes of stack that were scanned last GC cycle. Sourced from /gc/scan/stack:bytes. |
| `go_gc_scan_total_bytes` | gauge | The total amount space that is scannable. Sum of all metrics in /gc/scan. Sourced from /gc/scan/total:bytes. |
| `go_gc_stack_starting_size_bytes` | gauge | The stack size of new goroutines. Sourced from /gc/stack/starting-size:bytes. |
| `go_goroutines` | gauge | Number of goroutines that currently exist. |
| `go_info` | gauge | Information about the Go environment. |
| `go_memstats_alloc_bytes` | gauge | Number of bytes allocated in heap and currently in use. Equals to /memory/classes/heap/objects:bytes. |
| `go_memstats_alloc_bytes_total` | counter | Total number of bytes allocated in heap until now, even if released already. Equals to /gc/heap/allocs:bytes. |
| `go_memstats_buck_hash_sys_bytes` | gauge | Number of bytes used by the profiling bucket hash table. Equals to /memory/classes/profiling/buckets:bytes. |
| `go_memstats_frees_total` | counter | Total number of heap objects frees. Equals to /gc/heap/frees:objects + /gc/heap/tiny/allocs:objects. |
| `go_memstats_gc_sys_bytes` | gauge | Number of bytes used for garbage collection system metadata. Equals to /memory/classes/metadata/other:bytes. |
| `go_memstats_heap_alloc_bytes` | gauge | Number of heap bytes allocated and currently in use, same as go_memstats_alloc_bytes. Equals to /memory/classes/heap/objects:bytes. |
| `go_memstats_heap_idle_bytes` | gauge | Number of heap bytes waiting to be used. Equals to /memory/classes/heap/released:bytes + /memory/classes/heap/free:bytes. |
| `go_memstats_heap_inuse_bytes` | gauge | Number of heap bytes that are in use. Equals to /memory/classes/heap/objects:bytes + /memory/classes/heap/unused:bytes |
| `go_memstats_heap_objects` | gauge | Number of currently allocated objects. Equals to /gc/heap/objects:objects. |
| `go_memstats_heap_released_bytes` | gauge | Number of heap bytes released to OS. Equals to /memory/classes/heap/released:bytes. |
| `go_memstats_heap_sys_bytes` | gauge | Number of heap bytes obtained from system. Equals to /memory/classes/heap/objects:bytes + /memory/classes/heap/unused:bytes + /memory/classes/heap/released:bytes + /memory/classes/heap/free:bytes. |
| `go_memstats_last_gc_time_seconds` | gauge | Number of seconds since 1970 of last garbage collection. |
| `go_memstats_mallocs_total` | counter | Total number of heap objects allocated, both live and gc-ed. Semantically a counter version for go_memstats_heap_objects gauge. Equals to /gc/heap/allocs:objects + /gc/heap/tiny/allocs:objects. |
| `go_memstats_mcache_inuse_bytes` | gauge | Number of bytes in use by mcache structures. Equals to /memory/classes/metadata/mcache/inuse:bytes. |
| `go_memstats_mcache_sys_bytes` | gauge | Number of bytes used for mcache structures obtained from system. Equals to /memory/classes/metadata/mcache/inuse:bytes + /memory/classes/metadata/mcache/free:bytes. |
| `go_memstats_mspan_inuse_bytes` | gauge | Number of bytes in use by mspan structures. Equals to /memory/classes/metadata/mspan/inuse:bytes. |
| `go_memstats_mspan_sys_bytes` | gauge | Number of bytes used for mspan structures obtained from system. Equals to /memory/classes/metadata/mspan/inuse:bytes + /memory/classes/metadata/mspan/free:bytes. |
| `go_memstats_next_gc_bytes` | gauge | Number of heap bytes when next garbage collection will take place. Equals to /gc/heap/goal:bytes. |
| `go_memstats_other_sys_bytes` | gauge | Number of bytes used for other system allocations. Equals to /memory/classes/other:bytes. |
| `go_memstats_stack_inuse_bytes` | gauge | Number of bytes obtained from system for stack allocator in non-CGO environments. Equals to /memory/classes/heap/stacks:bytes. |
| `go_memstats_stack_sys_bytes` | gauge | Number of bytes obtained from system for stack allocator. Equals to /memory/classes/heap/stacks:bytes + /memory/classes/os-stacks:bytes. |
| `go_memstats_sys_bytes` | gauge | Number of bytes obtained from system. Equals to /memory/classes/total:byte. |
| `go_sched_gomaxprocs_threads` | gauge | The current runtime.GOMAXPROCS setting, or the number of operating system threads that can execute user-level Go code simultaneously. Sourced from /sched/gomaxprocs:threads. |
| `go_sched_goroutines_created_goroutines_total` | counter | Count of goroutines created since program start. Sourced from /sched/goroutines-created:goroutines. |
| `go_sched_goroutines_goroutines` | gauge | Count of live goroutines. Sourced from /sched/goroutines:goroutines. |
| `go_sched_goroutines_not_in_go_goroutines` | gauge | Approximate count of goroutines running or blocked in a system call or cgo call. Not guaranteed to add up to /sched/goroutines:goroutines with other goroutine metrics. Sourced from /sched/goroutines/not-in-go:goroutines. |
| `go_sched_goroutines_runnable_goroutines` | gauge | Approximate count of goroutines ready to execute, but not executing. Not guaranteed to add up to /sched/goroutines:goroutines with other goroutine metrics. Sourced from /sched/goroutines/runnable:goroutines. |
| `go_sched_goroutines_running_goroutines` | gauge | Approximate count of goroutines executing. Always less than or equal to /sched/gomaxprocs:threads. Not guaranteed to add up to /sched/goroutines:goroutines with other goroutine metrics. Sourced from /sched/goroutines/running:goroutines. |
| `go_sched_goroutines_waiting_goroutines` | gauge | Approximate count of goroutines waiting on a resource (I/O or sync primitives). Not guaranteed to add up to /sched/goroutines:goroutines with other goroutine metrics. Sourced from /sched/goroutines/waiting:goroutines. |
| `go_sched_latencies_seconds` | histogram | Distribution of the time goroutines have spent in the scheduler in a runnable state before actually running. Bucket counts increase monotonically. Sourced from /sched/latencies:seconds. |
| `go_sched_pauses_stopping_gc_seconds` | histogram | Distribution of individual GC-related stop-the-world stopping latencies. This is the time it takes from deciding to stop the world until all Ps are stopped. This is a subset of the total GC-related stop-the-world time (/sched/pauses/total/gc:seconds). During this time, some threads may be executing. Bucket counts increase monotonically. Sourced from /sched/pauses/stopping/gc:seconds. |
| `go_sched_pauses_stopping_other_seconds` | histogram | Distribution of individual non-GC-related stop-the-world stopping latencies. This is the time it takes from deciding to stop the world until all Ps are stopped. This is a subset of the total non-GC-related stop-the-world time (/sched/pauses/total/other:seconds). During this time, some threads may be executing. Bucket counts increase monotonically. Sourced from /sched/pauses/stopping/other:seconds. |
| `go_sched_pauses_total_gc_seconds` | histogram | Distribution of individual GC-related stop-the-world pause latencies. This is the time from deciding to stop the world until the world is started again. Some of this time is spent getting all threads to stop (this is measured directly in /sched/pauses/stopping/gc:seconds), during which some threads may still be running. Bucket counts increase monotonically. Sourced from /sched/pauses/total/gc:seconds. |
| `go_sched_pauses_total_other_seconds` | histogram | Distribution of individual non-GC-related stop-the-world pause latencies. This is the time from deciding to stop the world until the world is started again. Some of this time is spent getting all threads to stop (measured directly in /sched/pauses/stopping/other:seconds). Bucket counts increase monotonically. Sourced from /sched/pauses/total/other:seconds. |
| `go_sched_threads_total_threads` | gauge | The current count of live threads that are owned by the Go runtime. Sourced from /sched/threads/total:threads. |
| `go_sync_mutex_wait_total_seconds_total` | counter | Approximate cumulative time goroutines have spent blocked on a sync.Mutex, sync.RWMutex, or runtime-internal lock. This metric is useful for identifying global changes in lock contention. Collect a mutex or block profile using the runtime/pprof package for more detailed contention data. Sourced from /sync/mutex/wait/total:seconds. |
| `go_threads` | gauge | Number of OS threads created. |
| `net_conntrack_dialer_conn_attempted_total` | counter | Total number of connections attempted by the given dialer a given name. |
| `net_conntrack_dialer_conn_closed_total` | counter | Total number of connections closed which originated from the dialer of a given name. |
| `net_conntrack_dialer_conn_established_total` | counter | Total number of connections successfully established by the given dialer a given name. |
| `net_conntrack_dialer_conn_failed_total` | counter | Total number of connections failed to dial by the dialer a given name. |
| `net_conntrack_listener_conn_accepted_total` | counter | Total number of connections opened to the listener of a given name. |
| `net_conntrack_listener_conn_closed_total` | counter | Total number of connections closed that were made to the listener of a given name. |
| `process_cpu_seconds_total` | counter | Total user and system CPU time spent in seconds. |
| `process_max_fds` | gauge | Maximum number of open file descriptors. |
| `process_network_receive_bytes_total` | counter | Number of bytes received by the process over the network. |
| `process_network_transmit_bytes_total` | counter | Number of bytes sent by the process over the network. |
| `process_open_fds` | gauge | Number of open file descriptors. |
| `process_resident_memory_bytes` | gauge | Resident memory size in bytes. |
| `process_start_time_seconds` | gauge | Start time of the process since unix epoch in seconds. |
| `process_virtual_memory_bytes` | gauge | Virtual memory size in bytes. |
| `process_virtual_memory_max_bytes` | gauge | Maximum amount of virtual memory available in bytes. |
| `prometheus_api_notification_active_subscribers` | gauge | The current number of active notification subscribers. |
| `prometheus_api_notification_updates_dropped_total` | counter | Total number of notification updates dropped. |
| `prometheus_api_notification_updates_sent_total` | counter | Total number of notification updates sent. |
| `prometheus_build_info` | gauge | A metric with a constant '1' value labeled by version, revision, branch, goversion from which prometheus was built, and the goos and goarch for the build. |
| `prometheus_config_last_reload_success_timestamp_seconds` | gauge | Timestamp of the last successful configuration reload. |
| `prometheus_config_last_reload_successful` | gauge | Whether the last configuration reload attempt was successful. |
| `prometheus_engine_queries` | gauge | The current number of queries being executed or waiting. |
| `prometheus_engine_queries_concurrent_max` | gauge | The max number of concurrent queries. |
| `prometheus_engine_query_duration_histogram_seconds` | histogram | The duration of various parts of PromQL query execution. |
| `prometheus_engine_query_duration_seconds` | summary | Query timings |
| `prometheus_engine_query_log_enabled` | gauge | State of the query log. |
| `prometheus_engine_query_log_failures_total` | counter | The number of query log failures. |
| `prometheus_engine_query_samples_read_total` | counter | The total number of samples read by all queries (only new points per step for range-vector). |
| `prometheus_engine_query_samples_total` | counter | The total number of samples loaded by all queries (full window per step for range-vector). |
| `prometheus_http_request_duration_seconds` | histogram | Histogram of latencies for HTTP requests. |
| `prometheus_http_requests_total` | counter | Counter of HTTP requests. |
| `prometheus_http_response_size_bytes` | histogram | Histogram of response size for HTTP requests. |
| `prometheus_notifications_alertmanagers_discovered` | gauge | The number of alertmanagers discovered and active. |
| `prometheus_notifications_queue_capacity` | gauge | The capacity of the alert notifications queue. |
| `prometheus_ready` | gauge | Whether Prometheus startup was fully completed and the server is ready for normal operation. |
| `prometheus_remote_read_handler_queries` | gauge | The current number of remote read queries that are either in execution or queued on the handler. |
| `prometheus_remote_storage_exemplars_in_total` | counter | Exemplars in to remote storage, compare to exemplars out for queue managers. Deprecated, check prometheus_wal_watcher_records_read_total and prometheus_remote_storage_exemplars_dropped_total |
| `prometheus_remote_storage_highest_timestamp_in_seconds` | gauge | Highest timestamp that has come into the remote storage via the Appender interface, in seconds since epoch. Initialized to 0 when no data has been received yet. Deprecated, check prometheus_remote_storage_queue_highest_timestamp_seconds which is more accurate. |
| `prometheus_remote_storage_histograms_in_total` | counter | HistogramSamples in to remote storage, compare to histograms out for queue managers. Deprecated, check prometheus_wal_watcher_records_read_total and prometheus_remote_storage_histograms_dropped_total |
| `prometheus_remote_storage_samples_in_total` | counter | Samples in to remote storage, compare to samples out for queue managers. Deprecated, check prometheus_wal_watcher_records_read_total and prometheus_remote_storage_samples_dropped_total |
| `prometheus_remote_storage_string_interner_zero_reference_releases_total` | counter | The number of times release has been called for strings that are not interned. |
| `prometheus_rule_evaluation_duration_histogram_seconds` | histogram | The duration for a rule to execute. |
| `prometheus_rule_evaluation_duration_seconds` | summary | The duration for a rule to execute. |
| `prometheus_rule_evaluation_failures_total` | counter | The total number of rule evaluation failures. |
| `prometheus_rule_evaluations_total` | counter | The total number of rule evaluations. |
| `prometheus_rule_group_duration_histogram_seconds` | histogram | The duration of rule group evaluations. |
| `prometheus_rule_group_duration_seconds` | summary | The duration of rule group evaluations. |
| `prometheus_rule_group_interval_seconds` | gauge | The interval of a rule group. |
| `prometheus_rule_group_iterations_missed_total` | counter | The total number of rule group evaluations missed due to slow rule group evaluation. |
| `prometheus_rule_group_iterations_total` | counter | The total number of scheduled rule group evaluations, whether executed or missed. |
| `prometheus_rule_group_last_duration_seconds` | gauge | The duration of the last rule group evaluation. |
| `prometheus_rule_group_last_evaluation_samples` | gauge | The number of samples returned during the last rule group evaluation. |
| `prometheus_rule_group_last_evaluation_timestamp_seconds` | gauge | The timestamp of the last rule group evaluation in seconds. |
| `prometheus_rule_group_last_restore_duration_seconds` | gauge | The duration of the last alert rules alerts restoration using the `ALERTS_FOR_STATE` series. |
| `prometheus_rule_group_last_rule_duration_sum_seconds` | gauge | The sum of time in seconds it took to evaluate each rule in the group regardless of concurrency. This should be higher than the group duration if rules are evaluated concurrently. |
| `prometheus_rule_group_rules` | gauge | The number of rules. |
| `prometheus_sd_azure_cache_hit_total` | counter | Number of cache hit during refresh. |
| `prometheus_sd_azure_failures_total` | counter | Number of Azure service discovery refresh failures. |
| `prometheus_sd_consul_rpc_duration_seconds` | summary | The duration of a Consul RPC call in seconds. |
| `prometheus_sd_consul_rpc_failures_total` | counter | The number of Consul RPC call failures. |
| `prometheus_sd_discovered_targets` | gauge | Current number of discovered targets. |
| `prometheus_sd_dns_lookup_failures_total` | counter | The number of DNS-SD lookup failures. |
| `prometheus_sd_dns_lookups_total` | counter | The number of DNS-SD lookups. |
| `prometheus_sd_failed_configs` | gauge | Current number of service discovery configurations that failed to load. |
| `prometheus_sd_file_read_errors_total` | counter | The number of File-SD read errors. |
| `prometheus_sd_file_scan_duration_seconds` | summary | The duration of the File-SD scan in seconds. |
| `prometheus_sd_file_watcher_errors_total` | counter | The number of File-SD errors caused by filesystem watch failures. |
| `prometheus_sd_http_failures_total` | counter | Number of HTTP service discovery refresh failures. |
| `prometheus_sd_kubernetes_events_total` | counter | The number of Kubernetes events handled. |
| `prometheus_sd_kubernetes_failures_total` | counter | The number of failed WATCH/LIST requests. |
| `prometheus_sd_kuma_fetch_duration_seconds` | summary | The duration of a Kuma MADS fetch call. |
| `prometheus_sd_kuma_fetch_failures_total` | counter | The number of Kuma MADS fetch call failures. |
| `prometheus_sd_kuma_fetch_skipped_updates_total` | counter | The number of Kuma MADS fetch calls that result in no updates to the targets. |
| `prometheus_sd_last_update_timestamp_seconds` | gauge | Timestamp of the last update sent to the SD consumers. |
| `prometheus_sd_linode_failures_total` | counter | Number of Linode service discovery refresh failures. |
| `prometheus_sd_nomad_failures_total` | counter | Number of nomad service discovery refresh failures. |
| `prometheus_sd_received_updates_total` | counter | Total number of update events received from the SD providers. |
| `prometheus_sd_updates_delayed_total` | counter | Total number of update events that couldn't be sent immediately. |
| `prometheus_sd_updates_total` | counter | Total number of update events sent to the SD consumers. |
| `prometheus_target_interval_length_histogram_seconds` | histogram | Actual intervals between scrapes. |
| `prometheus_target_interval_length_seconds` | summary | Actual intervals between scrapes. |
| `prometheus_target_metadata_cache_bytes` | gauge | The number of bytes that are currently used for storing metric metadata in the cache |
| `prometheus_target_metadata_cache_entries` | gauge | Total number of metric metadata entries in the cache |
| `prometheus_target_scrape_duration_seconds` | histogram | Total duration of the scrape from start to commit completion in seconds. |
| `prometheus_target_scrape_pool_exceeded_label_limits_total` | counter | Total number of times scrape pools hit the label limits, during sync or config reload. |
| `prometheus_target_scrape_pool_exceeded_target_limit_total` | counter | Total number of times scrape pools hit the target limit, during sync or config reload. |
| `prometheus_target_scrape_pool_reloads_failed_total` | counter | Total number of failed scrape pool reloads. |
| `prometheus_target_scrape_pool_reloads_total` | counter | Total number of scrape pool reloads. |
| `prometheus_target_scrape_pool_symboltable_items` | gauge | Current number of symbols in table for this scrape pool. |
| `prometheus_target_scrape_pool_sync_total` | counter | Total number of syncs that were executed on a scrape pool. |
| `prometheus_target_scrape_pool_target_limit` | gauge | Maximum number of targets allowed in this scrape pool. |
| `prometheus_target_scrape_pool_targets` | gauge | Current number of targets in this scrape pool. |
| `prometheus_target_scrape_pools_failed_total` | counter | Total number of scrape pool creations that failed. |
| `prometheus_target_scrape_pools_total` | counter | Total number of scrape pool creation attempts. |
| `prometheus_target_scrapes_cache_flush_forced_total` | counter | How many times a scrape cache was flushed due to getting big while scrapes are failing. |
| `prometheus_target_scrapes_exceeded_body_size_limit_total` | counter | Total number of scrapes that hit the body size limit |
| `prometheus_target_scrapes_exceeded_native_histogram_bucket_limit_total` | counter | Total number of scrapes that hit the native histogram bucket limit and were rejected. |
| `prometheus_target_scrapes_exceeded_sample_limit_total` | counter | Total number of scrapes that hit the sample limit and were rejected. |
| `prometheus_target_scrapes_exemplar_out_of_order_total` | counter | Total number of exemplar rejected due to not being out of the expected order. |
| `prometheus_target_scrapes_sample_duplicate_timestamp_total` | counter | Total number of samples rejected due to duplicate timestamps but different values. |
| `prometheus_target_scrapes_sample_out_of_bounds_total` | counter | Total number of samples rejected due to timestamp falling outside of the time bounds. |
| `prometheus_target_scrapes_sample_out_of_order_total` | counter | Total number of samples rejected due to not being out of the expected order. |
| `prometheus_target_sync_failed_total` | counter | Total number of target sync failures. |
| `prometheus_target_sync_length_histogram_seconds` | histogram | Actual interval to sync the scrape pool. |
| `prometheus_target_sync_length_seconds` | summary | Actual interval to sync the scrape pool. |
| `prometheus_template_text_expansion_failures_total` | counter | The total number of template text expansion failures. |
| `prometheus_template_text_expansions_total` | counter | The total number of template text expansions. |
| `prometheus_treecache_watcher_goroutines` | gauge | The current number of watcher goroutines. |
| `prometheus_treecache_zookeeper_failures_total` | counter | The total number of ZooKeeper failures. |
| `prometheus_tsdb_blocks_loaded` | gauge | Number of currently loaded data blocks |
| `prometheus_tsdb_checkpoint_creations_failed_total` | counter | Total number of checkpoint creations that failed. |
| `prometheus_tsdb_checkpoint_creations_total` | counter | Total number of checkpoint creations attempted. |
| `prometheus_tsdb_checkpoint_deletions_failed_total` | counter | Total number of checkpoint deletions that failed. |
| `prometheus_tsdb_checkpoint_deletions_total` | counter | Total number of checkpoint deletions attempted. |
| `prometheus_tsdb_clean_start` | gauge | -1: lockfile is disabled. 0: a lockfile from a previous execution was replaced. 1: lockfile creation was clean |
| `prometheus_tsdb_compaction_chunk_range_seconds` | histogram | Final time range of chunks on their first compaction |
| `prometheus_tsdb_compaction_chunk_samples` | histogram | Final number of samples on their first compaction |
| `prometheus_tsdb_compaction_chunk_size_bytes` | histogram | Final size of chunks on their first compaction |
| `prometheus_tsdb_compaction_duration_seconds` | histogram | Duration of compaction runs |
| `prometheus_tsdb_compaction_populating_block` | gauge | Set to 1 when a block is currently being written to the disk. |
| `prometheus_tsdb_compactions_failed_total` | counter | Total number of compactions that failed for the partition. |
| `prometheus_tsdb_compactions_skipped_total` | counter | Total number of skipped compactions due to disabled auto compaction. |
| `prometheus_tsdb_compactions_total` | counter | Total number of compactions that were executed for the partition. |
| `prometheus_tsdb_compactions_triggered_total` | counter | Total number of triggered compactions for the partition. |
| `prometheus_tsdb_data_replay_duration_seconds` | gauge | Time taken to replay the data on disk. |
| `prometheus_tsdb_exemplar_exemplars_appended_total` | counter | Total number of appended exemplars. |
| `prometheus_tsdb_exemplar_exemplars_in_storage` | gauge | Number of exemplars currently in circular storage. |
| `prometheus_tsdb_exemplar_last_exemplars_timestamp_seconds` | gauge | The timestamp of the oldest exemplar stored in circular storage. Useful to check for what timerange the current exemplar buffer limit allows. This usually means the last timestampfor all exemplars for a typical setup. This is not true though if one of the series timestamp is in future compared to rest series. |
| `prometheus_tsdb_exemplar_max_exemplars` | gauge | Total number of exemplars the exemplar storage can store, resizeable. |
| `prometheus_tsdb_exemplar_out_of_order_exemplars_total` | counter | Total number of out of order exemplar ingestion failed attempts. |
| `prometheus_tsdb_exemplar_series_with_exemplars_in_storage` | gauge | Number of series with exemplars currently in circular storage. |
| `prometheus_tsdb_head_active_appenders` | gauge | Number of currently active appender transactions |
| `prometheus_tsdb_head_chunks` | gauge | Total number of chunks in the head block. |
| `prometheus_tsdb_head_chunks_created_total` | counter | Total number of chunks created in the head |
| `prometheus_tsdb_head_chunks_removed_total` | counter | Total number of chunks removed in the head |
| `prometheus_tsdb_head_chunks_storage_size_bytes` | gauge | Size of the chunks_head directory. |
| `prometheus_tsdb_head_gc_duration_seconds` | summary | Runtime of garbage collection in the head block. |
| `prometheus_tsdb_head_max_time` | gauge | Maximum timestamp of the head block. The unit is decided by the library consumer. |
| `prometheus_tsdb_head_max_time_seconds` | gauge | Maximum timestamp of the head block. |
| `prometheus_tsdb_head_min_time` | gauge | Minimum time bound of the head block. The unit is decided by the library consumer. |
| `prometheus_tsdb_head_min_time_seconds` | gauge | Minimum time bound of the head block. |
| `prometheus_tsdb_head_native_histogram_buckets` | gauge | Number of positive and negative bucket entries in the most recent in-order native histogram sample of each series, including entries carried by staleness markers. |
| `prometheus_tsdb_head_native_histogram_series` | gauge | Number of series whose most recent in-order sample is a native histogram. |
| `prometheus_tsdb_head_out_of_order_samples_appended_total` | counter | Total number of appended out of order samples. |
| `prometheus_tsdb_head_samples_appended_total` | counter | Total number of appended samples. |
| `prometheus_tsdb_head_series` | gauge | Total number of series in the head block. |
| `prometheus_tsdb_head_series_created_total` | counter | Total number of series created in the head |
| `prometheus_tsdb_head_series_not_found_total` | counter | Total number of requests for series that were not found. |
| `prometheus_tsdb_head_series_removed_total` | counter | Total number of series removed in the head |
| `prometheus_tsdb_head_stale_series` | gauge | Total number of stale series in the head block. |
| `prometheus_tsdb_head_truncations_failed_total` | counter | Total number of head truncations that failed. |
| `prometheus_tsdb_head_truncations_total` | counter | Total number of head truncations attempted. |
| `prometheus_tsdb_isolation_high_watermark` | gauge | The highest TSDB append ID that has been given out. |
| `prometheus_tsdb_isolation_low_watermark` | gauge | The lowest TSDB append ID that is still referenced. |
| `prometheus_tsdb_lowest_timestamp` | gauge | Lowest timestamp value stored in the database. The unit is decided by the library consumer. |
| `prometheus_tsdb_lowest_timestamp_seconds` | gauge | Lowest timestamp value stored in the database. |
| `prometheus_tsdb_mmap_chunk_corruptions_total` | counter | Total number of memory-mapped chunk corruptions. |
| `prometheus_tsdb_mmap_chunks_total` | counter | Total number of chunks that were memory-mapped. |
| `prometheus_tsdb_out_of_bound_samples_total` | counter | Total number of out of bound samples ingestion failed attempts with out of order support disabled. |
| `prometheus_tsdb_out_of_order_samples_total` | counter | Total number of out of order samples ingestion failed attempts due to out of order being disabled. |
| `prometheus_tsdb_reloads_failures_total` | counter | Number of times the database failed to reloadBlocks block data from disk. |
| `prometheus_tsdb_reloads_total` | counter | Number of times the database reloaded block data from disk. |
| `prometheus_tsdb_retention_limit_bytes` | gauge | Max number of bytes to be retained in the tsdb blocks, configured 0 means disabled |
| `prometheus_tsdb_retention_limit_percentage` | gauge | Max percentage of total storage space to be retained in the tsdb blocks, configured 0 means disabled |
| `prometheus_tsdb_retention_limit_seconds` | gauge | How long to retain samples in storage. |
| `prometheus_tsdb_sample_ooo_delta` | histogram | Delta in seconds by which a sample is considered out of order (reported regardless of OOO time window and whether sample is accepted or not). |
| `prometheus_tsdb_selected_series_compaction_duration_seconds` | histogram | Duration of compactions triggered for an explicit caller-provided list of series references. |
| `prometheus_tsdb_selected_series_compactions_failed_total` | counter | Total number of compactions triggered for an explicit caller-provided list of series references that failed. |
| `prometheus_tsdb_selected_series_compactions_triggered_total` | counter | Total number of compactions triggered for an explicit caller-provided list of series references. |
| `prometheus_tsdb_size_retentions_total` | counter | The number of times that blocks were deleted because the maximum number of bytes was exceeded. |
| `prometheus_tsdb_snapshot_replay_error_total` | counter | Total number snapshot replays that failed. |
| `prometheus_tsdb_stale_series_compaction_duration_seconds` | histogram | Duration of stale series compaction runs. |
| `prometheus_tsdb_stale_series_compactions_failed_total` | counter | Total number of stale series compactions that failed. |
| `prometheus_tsdb_stale_series_compactions_triggered_total` | counter | Total number of triggered stale series compactions. |
| `prometheus_tsdb_storage_blocks_bytes` | gauge | The number of bytes that are currently used for local storage by all blocks. |
| `prometheus_tsdb_symbol_table_size_bytes` | gauge | Size of symbol table in memory for loaded blocks |
| `prometheus_tsdb_time_retentions_total` | counter | The number of times that blocks were deleted because the maximum time limit was exceeded. |
| `prometheus_tsdb_tombstone_cleanup_seconds` | histogram | The time taken to recompact blocks to remove tombstones. |
| `prometheus_tsdb_too_old_samples_total` | counter | Total number of out of order samples ingestion failed attempts with out of support enabled, but sample outside of time window. |
| `prometheus_tsdb_vertical_compactions_total` | counter | Total number of compactions done on overlapping blocks. |
| `prometheus_tsdb_wal_completed_pages_total` | counter | Total number of completed pages. |
| `prometheus_tsdb_wal_corruptions_total` | counter | Total number of WAL corruptions. |
| `prometheus_tsdb_wal_fsync_duration_seconds` | summary | Duration of write log fsync. |
| `prometheus_tsdb_wal_page_flushes_total` | counter | Total number of page flushes. |
| `prometheus_tsdb_wal_record_bytes_saved_total` | counter | Total number of bytes saved by the optional record compression. Use this metric to learn about the effectiveness compression. |
| `prometheus_tsdb_wal_record_part_writes_total` | counter | Total number of record parts written before flushing. |
| `prometheus_tsdb_wal_record_parts_bytes_written_total` | counter | Total number of record part bytes written before flushing, including CRC and compression headers. |
| `prometheus_tsdb_wal_segment_current` | gauge | Write log segment index that TSDB is currently writing to. |
| `prometheus_tsdb_wal_storage_size_bytes` | gauge | Size of the write log directory. |
| `prometheus_tsdb_wal_truncate_duration_seconds` | summary | Duration of WAL truncation. |
| `prometheus_tsdb_wal_truncations_failed_total` | counter | Total number of write log truncations that failed. |
| `prometheus_tsdb_wal_truncations_total` | counter | Total number of write log truncations attempted. |
| `prometheus_tsdb_wal_writes_failed_total` | counter | Total number of write log writes that failed. |
| `prometheus_web_federation_errors_total` | counter | Total number of errors that occurred while sending federation responses. |
| `prometheus_web_federation_warnings_total` | counter | Total number of warnings that occurred while sending federation responses. |
| `promhttp_metric_handler_requests_in_flight` | gauge | Current number of scrapes being served. |
| `promhttp_metric_handler_requests_total` | counter | Total number of scrapes by HTTP status code. |

## node_exporter

| Metric | Type | Description |
|--------|------|-------------|
| `node_cpu_seconds_total` | counter | Seconds the CPUs spent in each mode. |
| `node_disk_read_bytes_total` | counter | The total number of bytes read successfully. |
| `node_disk_written_bytes_total` | counter | The total number of bytes written successfully. |
| `node_filesystem_avail_bytes` | gauge | Filesystem space available to non-root users. |
| `node_filesystem_size_bytes` | gauge | Filesystem size in bytes. |
| `node_memory_Active_anon_bytes` | gauge | Memory information field Active_anon. |
| `node_memory_Active_bytes` | gauge | Memory information field Active. |
| `node_memory_Active_file_bytes` | gauge | Memory information field Active_file. |
| `node_memory_AnonHugePages_bytes` | gauge | Memory information field AnonHugePages. |
| `node_memory_AnonPages_bytes` | gauge | Memory information field AnonPages. |
| `node_memory_Bounce_bytes` | gauge | Memory information field Bounce. |
| `node_memory_Buffers_bytes` | gauge | Memory information field Buffers. |
| `node_memory_Cached_bytes` | gauge | Memory information field Cached. |
| `node_memory_CommitLimit_bytes` | gauge | Memory information field CommitLimit. |
| `node_memory_Committed_AS_bytes` | gauge | Memory information field Committed_AS. |
| `node_memory_DirectMap1G_bytes` | gauge | Memory information field DirectMap1G. |
| `node_memory_DirectMap2M_bytes` | gauge | Memory information field DirectMap2M. |
| `node_memory_DirectMap4k_bytes` | gauge | Memory information field DirectMap4k. |
| `node_memory_Dirty_bytes` | gauge | Memory information field Dirty. |
| `node_memory_FileHugePages_bytes` | gauge | Memory information field FileHugePages. |
| `node_memory_FilePmdMapped_bytes` | gauge | Memory information field FilePmdMapped. |
| `node_memory_HardwareCorrupted_bytes` | gauge | Memory information field HardwareCorrupted. |
| `node_memory_HugePages_Free_bytes` | gauge | Memory information field HugePages_Free. |
| `node_memory_HugePages_Rsvd_bytes` | gauge | Memory information field HugePages_Rsvd. |
| `node_memory_HugePages_Surp_bytes` | gauge | Memory information field HugePages_Surp. |
| `node_memory_HugePages_Total_bytes` | gauge | Memory information field HugePages_Total. |
| `node_memory_Hugepagesize_bytes` | gauge | Memory information field Hugepagesize. |
| `node_memory_Hugetlb_bytes` | gauge | Memory information field Hugetlb. |
| `node_memory_Inactive_anon_bytes` | gauge | Memory information field Inactive_anon. |
| `node_memory_Inactive_bytes` | gauge | Memory information field Inactive. |
| `node_memory_Inactive_file_bytes` | gauge | Memory information field Inactive_file. |
| `node_memory_KReclaimable_bytes` | gauge | Memory information field KReclaimable. |
| `node_memory_KernelStack_bytes` | gauge | Memory information field KernelStack. |
| `node_memory_Mapped_bytes` | gauge | Memory information field Mapped. |
| `node_memory_MemAvailable_bytes` | gauge | Memory information field MemAvailable. |
| `node_memory_MemFree_bytes` | gauge | Memory information field MemFree. |
| `node_memory_MemTotal_bytes` | gauge | Memory information field MemTotal. |
| `node_memory_Mlocked_bytes` | gauge | Memory information field Mlocked. |
| `node_memory_NFS_Unstable_bytes` | gauge | Memory information field NFS_Unstable. |
| `node_memory_PageTables_bytes` | gauge | Memory information field PageTables. |
| `node_memory_Percpu_bytes` | gauge | Memory information field Percpu. |
| `node_memory_SReclaimable_bytes` | gauge | Memory information field SReclaimable. |
| `node_memory_SUnreclaim_bytes` | gauge | Memory information field SUnreclaim. |
| `node_memory_SecPageTables_bytes` | gauge | Memory information field SecPageTables. |
| `node_memory_ShmemHugePages_bytes` | gauge | Memory information field ShmemHugePages. |
| `node_memory_ShmemPmdMapped_bytes` | gauge | Memory information field ShmemPmdMapped. |
| `node_memory_Shmem_bytes` | gauge | Memory information field Shmem. |
| `node_memory_Slab_bytes` | gauge | Memory information field Slab. |
| `node_memory_SwapCached_bytes` | gauge | Memory information field SwapCached. |
| `node_memory_SwapFree_bytes` | gauge | Memory information field SwapFree. |
| `node_memory_SwapTotal_bytes` | gauge | Memory information field SwapTotal. |
| `node_memory_Unaccepted_bytes` | gauge | Memory information field Unaccepted. |
| `node_memory_Unevictable_bytes` | gauge | Memory information field Unevictable. |
| `node_memory_VmallocChunk_bytes` | gauge | Memory information field VmallocChunk. |
| `node_memory_VmallocTotal_bytes` | gauge | Memory information field VmallocTotal. |
| `node_memory_VmallocUsed_bytes` | gauge | Memory information field VmallocUsed. |
| `node_memory_WritebackTmp_bytes` | gauge | Memory information field WritebackTmp. |
| `node_memory_Writeback_bytes` | gauge | Memory information field Writeback. |
| `node_memory_Zswap_bytes` | gauge | Memory information field Zswap. |
| `node_memory_Zswapped_bytes` | gauge | Memory information field Zswapped. |
| `node_network_receive_bytes_total` | counter | Network device statistic receive_bytes. |
| `node_network_transmit_bytes_total` | counter | Network device statistic transmit_bytes. |
