# Metrics & Observability

A Prometheus sidecar captures the full stack for post-hoc exploration and
live dashboarding via Grafana.

---

## Port map

### Single-GPU stack (1 lmcache + 1 vLLM)

| Service | Port | Protocol | Notes |
| --- | --- | --- | --- |
| vLLM OpenAI API + `/metrics` | **8000** | HTTP | 8001 = vram-only baseline arm |
| LMCache HTTP API + `/metrics` | **8080** | HTTP | OTel Prometheus fallback; `lmcache_mp_*` metrics |
| NIXL telemetry `/metrics` | **19090** | HTTP | Native NIXL exporter; `agent_*` metrics |
| LMCache ZMQ connector | **6555** | TCP | `LMCacheMPConnector` vLLM↔lmcache channel |
| LMCache coordinator REST API | **9300** | HTTP | `/instances`, `/directory/stats`, `/quota` |
| LMCache coordinator `/metrics` | **9301** | HTTP | Prometheus exporter sidecar in coordinator container |
| Prometheus | **9090** | HTTP | TSDB written to `AIC_METRICS_DIR` |
| Grafana | **3000** | HTTP | `AIC_GRAFANA_PORT` to override |
| node-exporter | **9100** | HTTP | CPU/mem/disk/NVMe/InfiniBand |
| nvme-exporter | **9998** | HTTP | NVMe SMART + health |
| rdma-exporter | **9879** | HTTP | RDMA/RoCE sysfs counters |
| amdgpu-exporter | **5000** | HTTP | `gpu_*` metrics; 5050 = fallback |
| hsa-snoop | **9488** | HTTP | HSA AQL + AIS P2P storage |

### Multi-GPU scaling (N GPU pairs)

Each lmcache+vLLM pair requires a unique set of ports. With GPU index N
(0-based), the pattern is:

| Port | GPU 0 | GPU 1 | GPU N |
| --- | --- | --- | --- |
| vLLM OpenAI API | 8000 | 8001 | `8000 + N` |
| lmcache HTTP API | 8080 | 8081 | `8080 + N` |
| lmcache ZMQ | 6555 | 6556 | `6555 + N` |
| NIXL telemetry | 19090 | 19091 | `19090 + N` |

The coordinator, Prometheus, Grafana, and all exporters are **shared
across the fleet** — one instance per node regardless of GPU count.

The upstream two-server branch demonstrates this pattern with `GPU2`,
`LMCACHE_PORT_2`, `LMCACHE_HTTP_PORT_2`, `VLLM_PORT_2` env vars and
Docker Compose `extends:`. Beyond 2 GPU pairs a templated approach is
needed (per-GPU override files or a generator script).

---

## Grafana dashboard

Pre-provisioned at `http://localhost:3000` (anonymous viewer, no login).

```bash
make up HF_TOKEN=hf_... NVME_DATA=/mnt/lmcache-nvme NFS_DATA=/mnt/lmcache-nfs \
    VLLM_MODEL=openai/gpt-oss-120b
# → open http://localhost:3000
```

| Variable | Default | Description |
| --- | --- | --- |
| `AIC_GRAFANA_PORT` | `3000` | Host port Grafana listens on |
| `AIC_GRAFANA_IMAGE` | `grafana/grafana:13.1.3` | Grafana image |
| `AIC_PROM_IMAGE` | `prom/prometheus:v3.13.2` | Prometheus image |

The **AIC Overview** dashboard covers eleven rows:

| Row | Key panels |
| --- | --- |
| **Summary** | Cumulative token counters (input, output, computed, L1/L2 hits), TTFT/TPOT, total tokens/s, cache hit rates |
| **GPU** | VRAM %, power, clocks, temperatures, GFX/UMC/MMA activity, PCIe / xGMI fabric speed |
| **vLLM** | Throughput, TTFT/ITL/E2E latency p50+p95, KV cache usage %, prefix cache hit rate, preemptions |
| **LMCache** | L1 usage gauge, L1/L2 chunk rates, L0→L1 throughput, reuse gap, in-flight L2 stores |
| **NIXL / L2** | TX/RX throughput (B/s), average transfer latency (µs/req) |
| **NVMe** | Throughput, IOPS, latency, capacity used, IO pressure (PSI) |
| **Node / RDMA** | CPU load, memory, filesystem space, IB/RoCE port throughput |
| **hsa-snoop** | Kernel launch rate by kernel/process, duration histogram, SDMA throughput |
| **LMCache Coordinator** | Up/down status, registered instances, key directory size, event stream gaps, L2 quota |
| **LMCache — Extended** | L0→L1 load throughput, L2 in-flight queue depth, chunk reuse gap, HTTP latency |
| **LMCache — Adapter & Transport Health** | L2 adapter state (active/draining), NIXL transfer latency, L1 allocation failures |

Dashboard JSON: [`monitoring/grafana/dashboards/aic-overview.json`](../monitoring/grafana/dashboards/aic-overview.json).
Edits are provisioned from disk within 60 seconds without restarting the container.

**Scrape intervals:** global 15 s / timeout 10 s. lmcache `/metrics` is
served by the same FastAPI process as KV requests and blocks under heavy
load; the 15 s interval prevents gaps in time-series panels.

---

## Prometheus scrape jobs

| Job | Target | Interval |
| --- | --- | --- |
| `vllm` | `aic-vllm-gpu0:8000`, `:8001` | 15 s (global) |
| `lmcache` | `aic-lmcache:8080` | 15 s (global) |
| `lmcache_coordinator` | `aic-coordinator-exporter:9301` | 15 s (global) |
| `nixl` | `aic-lmcache:19090` | 15 s (global) |
| `node_exporter` | `aic-node-exporter:9100` | 15 s (global) |
| `nvme_exporter` | `aic-nvme-exporter:9998` | 15 s (global) |
| `rdma_exporter` | `aic-rdma-exporter:9879` | 15 s (global) |
| `amd_metrics_exporter` | `aic-amdgpu-exporter:5000`, `:5050` | 15 s (global) |
| `hsa_snoop` | `host-gateway:9488` | 15 s (global) |

---

## Metric reference

Metrics are grouped by Prometheus scrape job. Histogram families are
listed once without `_bucket`/`_count`/`_sum` suffixes. Metrics marked
*lazy* only appear after KV traffic flows — they are registered by the
OTel exporter on first observation.

### `amd_metrics_exporter` — AMD GPU (`gpu_*`)

| Metric | Type | Description |
| --- | --- | --- |
| `gpu_average_package_power` | gauge | Socket power (W) |
| `gpu_clock` | gauge | Current clock frequency (MHz); `clock_type` label: `system`, `memory`, `fabric`, `soc`, `video`, `dce`, `data` |
| `gpu_edge_temperature` | gauge | Edge temperature (°C) |
| `gpu_junction_temperature` | gauge | Junction/hotspot temperature (°C) |
| `gpu_memory_temperature` | gauge | Memory temperature (°C) |
| `gpu_gfx_activity` | gauge | Graphics engine utilisation (0–100 %) |
| `gpu_umc_activity` | gauge | Memory controller utilisation (0–100 %) |
| `gpu_mma_activity` | gauge | Multimedia engine utilisation (0–100 %) |
| `gpu_vcn_activity` | gauge | VCN encode/decode utilisation (%) |
| `gpu_total_vram` | gauge | Total VRAM (MB) |
| `gpu_used_vram` | gauge | Used VRAM (MB) |
| `gpu_free_vram` | gauge | Free VRAM (MB) |
| `gpu_total_visible_vram` | gauge | Total visible VRAM (MB) |
| `gpu_used_visible_vram` | gauge | Used visible VRAM (MB) |
| `gpu_free_visible_vram` | gauge | Free visible VRAM (MB) |
| `gpu_total_gtt` | gauge | Total GTT memory (MB) |
| `gpu_used_gtt` | gauge | Used GTT memory (MB) |
| `gpu_free_gtt` | gauge | Free GTT memory (MB) |
| `gpu_gfx_voltage` | gauge | GFX voltage (mV) |
| `gpu_memory_voltage` | gauge | Memory voltage (mV) |
| `gpu_voltage` | gauge | SoC voltage (mV) |
| `gpu_max_clock` | gauge | Max clock frequency (MHz) |
| `gpu_min_clock` | gauge | Min clock frequency (MHz) |
| `gpu_health` | gauge | GPU health (0 = unhealthy, 1 = healthy) |
| `gpu_nodes_total` | gauge | Number of GPUs in node |
| `gpu_xgmi_nbr_N_tx_thrput` | gauge | xGMI fabric link N outbound beat throughput (32 B/beat) |
| `gpu_xgmi_nbr_N_beats_tx` | gauge | xGMI data beats sent to neighbor N |
| `gpu_ecc_correct_*` | gauge | Correctable ECC errors per block |
| `gpu_ecc_uncorrect_*` | gauge | Uncorrectable ECC errors per block |
| `gpu_ecc_deferred_*` | gauge | Accumulated deferred ECC errors per block |
| `pcie_speed` | gauge | Current PCIe link speed (GT/s) |
| `pcie_max_speed` | gauge | Maximum PCIe link speed (GT/s) |

### `hsa_snoop` — HSA kernels + AIS storage (`:9488`)

| Metric | Type | Description |
| --- | --- | --- |
| `hsa_kernel_launches_total` | counter | HSA AQL kernel dispatches; `kernel_name` label |
| `hsa_kernel_duration_seconds` | histogram | Per-kernel execution time |
| `hsa_active_queues` | gauge | Active AQL queues |
| `hsa_active_sdma_queues` | gauge | Active SDMA queues |
| `hsa_barrier_packets_total` | counter | AQL barrier packets; `comm` (process) label |
| `hsa_sdma_bytes_total` | counter | SDMA DMA bytes transferred |
| `hsa_sdma_copies_total` | counter | SDMA copy operations |
| `hsa_sdma_packets_total` | counter | SDMA packets |
| `ais_active` | gauge | 1 if AIS P2P I/O has been observed (latches high) |
| `ais_rx_ops_total` | counter | AIS receive operations |
| `ais_tx_ops_total` | counter | AIS transmit operations |
| `ais_rx_bytes_total` | counter | AIS bytes received |
| `ais_tx_bytes_total` | counter | AIS bytes transmitted |
| `ais_rx_errors_total` | counter | AIS receive errors |
| `ais_tx_errors_total` | counter | AIS transmit errors |
| `ais_rx_latency_seconds` | histogram | AIS receive latency |
| `ais_tx_latency_seconds` | histogram | AIS transmit latency |

> hsa-snoop holds tracefs exclusively on the host. It runs as a host-side
> service or via the `exporters` monitoring compose profile. Prometheus
> reaches it at `host-gateway:9488`.

### `lmcache` — LMCache MP server (`:8080`)

LMCache uses OpenTelemetry with a Prometheus fallback at `--http-port`
(default 8080). Metrics marked *lazy* only appear after KV traffic.

#### Event bus

| Metric | Type | Description |
| --- | --- | --- |
| `lmcache_mp_event_bus_queue_depth` | gauge | Events queued in the internal EventBus |
| `lmcache_mp_event_bus_drain_lag_seconds` | gauge | Age of oldest queued event; rising = drain thread falling behind |
| `lmcache_mp_event_bus_dropped_events_total` | counter | Events dropped because the queue was full |

#### L1 DRAM cache

| Metric | Type | Description |
| --- | --- | --- |
| `lmcache_mp_l1_memory_usage_bytes` | gauge | Bytes currently in L1 DRAM pool |
| `lmcache_mp_l1_usage_ratio` | gauge | L1 fill fraction (0.0–1.0) |
| `lmcache_mp_l1_write_chunks_total` | counter | *lazy* KV chunks written to L1 |
| `lmcache_mp_l1_read_chunks_total` | counter | *lazy* KV chunks read from L1 |
| `lmcache_mp_l1_evicted_chunks_total` | counter | *lazy* L1 chunks evicted (to L2 or dropped) |
| `lmcache_mp_l1_allocation_failure_chunks_total` | counter | *lazy* Chunks dropped because L1 was full and no L2 available |
| `lmcache_mp_l1_eviction_loop_ticks_total` | counter | L1 LRU eviction loop iterations |
| `lmcache_mp_l1_eviction_loop_triggered_total` | counter | *lazy* Times eviction fired above the watermark |
| `lmcache_mp_l1_chunk_lifetime_seconds` | histogram | *lazy* Time a chunk spends in L1 |
| `lmcache_mp_l1_chunk_idle_before_evict_seconds` | histogram | *lazy* Idle time before L1 eviction |
| `lmcache_mp_l1_chunk_reuse_gap_seconds` | histogram | *lazy* Time between writes and reads of the same chunk |
| `lmcache_mp_l1_chunk_evict_reuse_gap_seconds` | histogram | *lazy* Gap between eviction and re-request |

#### L0 (GPU) ↔ L1 (DRAM) transfers

| Metric | Type | Description |
| --- | --- | --- |
| `lmcache_mp_l0_l1_store_throughput_GB_per_second` | histogram | *lazy* GPU→DRAM KV store throughput (GB/s) |
| `lmcache_mp_l0_l1_load_throughput_GB_per_second` | histogram | *lazy* DRAM→GPU KV load throughput (GB/s) |
| `lmcache_mp_l0_block_lifetime_seconds` | histogram | *lazy* GPU KV block lifetime before eviction |
| `lmcache_mp_l0_block_idle_before_evict_seconds` | histogram | *lazy* GPU block idle time before eviction |
| `lmcache_mp_l0_block_reuse_gap_seconds` | histogram | *lazy* Gap between GPU block reuse events |

#### L2 (NVMe / NFS) store

| Metric | Type | Description |
| --- | --- | --- |
| `lmcache_mp_l2_usage_bytes` | gauge | Bytes in the L2 pool; `l2_name` label |
| `lmcache_mp_l2_store_adapters` | gauge | L2 store adapters by `state` (`active`, `draining`) |
| `lmcache_mp_l2_store_submitted_requests_total` | counter | *lazy* L2 store requests submitted |
| `lmcache_mp_l2_store_submitted_objects_chunks_total` | counter | *lazy* Chunks submitted for L2 store |
| `lmcache_mp_l2_store_completed_requests_total` | counter | *lazy* L2 store requests completed; `l2_name` label |
| `lmcache_mp_l2_store_completed_objects_chunks_total` | counter | *lazy* Chunks successfully stored to L2 |
| `lmcache_mp_l2_store_throughput_GB_per_second` | histogram | *lazy* L2 store throughput (GB/s); `l2_name` label |
| `lmcache_mp_l2_evicted_objects_chunks_total` | counter | *lazy* Chunks evicted from L2 |
| `lmcache_mp_num_inflight_l2_stores` | gauge | In-flight L2 store operations; `adapter_index`+`l2_name` labels |
| `lmcache_mp_num_inflight_l2_loads` | gauge | In-flight L2 load operations |

#### L2 prefetch

| Metric | Type | Description |
| --- | --- | --- |
| `lmcache_mp_l2_prefetch_adapters` | gauge | L2 prefetch adapters by `state` |
| `lmcache_mp_l2_prefetch_lookup_requests_total` | counter | *lazy* L2 prefetch lookups attempted |
| `lmcache_mp_l2_prefetch_lookup_objects_chunks_total` | counter | *lazy* Chunks looked up in L2 |
| `lmcache_mp_l2_prefetch_hit_chunks_total` | counter | *lazy* Chunks successfully retrieved from L2 |
| `lmcache_mp_l2_prefetch_load_submitted_requests_total` | counter | *lazy* L2 load tasks submitted |
| `lmcache_mp_l2_prefetch_load_submitted_objects_chunks_total` | counter | *lazy* Chunks submitted for load |
| `lmcache_mp_l2_prefetch_load_completed_chunks_total` | counter | *lazy* Chunks successfully loaded from L2 |
| `lmcache_mp_l2_load_completed_requests_total` | counter | *lazy* L2 load requests completed |
| `lmcache_mp_l2_load_throughput_GB_per_second` | histogram | *lazy* L2 load throughput (GB/s) |
| `lmcache_mp_active_prefetch_jobs` | gauge | Active prefetch jobs |

#### Prefetch hit/miss (combined L1+L2)

| Metric | Type | Description |
| --- | --- | --- |
| `lmcache_mp_prefetch_hit_chunks_total` | counter | *lazy* Cache hits by `tier` label (`l1` or `l2`) |
| `lmcache_mp_prefetch_miss_chunks_total` | counter | *lazy* Cache misses |
| `lmcache_mp_prefetch_requests_total` | counter | *lazy* Total prefetch lookup requests |
| `lmcache_mp_active_p2p_lookup_jobs` | gauge | Active P2P (cross-instance) lookup jobs |

#### Reuse gap / effectiveness

| Metric | Type | Description |
| --- | --- | --- |
| `lmcache_mp_real_reuse_gap_seconds` | histogram | *lazy* Time between a chunk being stored and its next cache hit |
| `lmcache_mp_real_reuse_gap_objects` | histogram | *lazy* Per-object reuse gap |

#### Token-level lookup

| Metric | Type | Description |
| --- | --- | --- |
| `lmcache_mp_lookup_requested_tokens_total` | counter | *lazy* Tokens requested from cache |
| `lmcache_mp_lookup_hit_tokens_total` | counter | *lazy* Tokens served from cache |
| `lmcache_mp_num_chunks_loaded_total` | counter | *lazy* Total chunks loaded back to GPU |
| `lmcache_mp_inflight_load_memory_usage_bytes` | gauge | Memory held by in-flight load operations |

### `lmcache_coordinator` — MP coordinator exporter (`:9301`)

Custom Prometheus exporter that polls the coordinator REST API. All
metrics use `job="lmcache_coordinator"`.

| Metric | Type | Description |
| --- | --- | --- |
| `lmcache_coordinator_up` | gauge | 1 if `/healthz` reports healthy |
| `lmcache_coordinator_instances_registered` | gauge | Live MP servers in the registry |
| `lmcache_coordinator_instance_info` | gauge | Per-instance metadata (labels: `instance_id`, `ip`, `http_port`) |
| `lmcache_coordinator_directory_keys_total` | gauge | Unique KV chunk keys with at least one placement |
| `lmcache_coordinator_directory_placements_total` | gauge | Total placements across all keys and backends |
| `lmcache_coordinator_instance_l1_keys` | gauge | L1 keys reported per instance via event stream |
| `lmcache_coordinator_instance_seq_gap` | gauge | 1 if a sequence gap was detected in the instance's event stream |
| `lmcache_coordinator_quota_usage_bytes_total` | gauge | Total L2 byte usage tracked across all cache salts |
| `lmcache_coordinator_quota_usage_bytes` | gauge | L2 byte usage per `cache_salt` |
| `lmcache_coordinator_quota_limit_bytes` | gauge | L2 quota limit per `cache_salt`; −1 = unlimited |
| `lmcache_coordinator_scrape_duration_seconds` | gauge | Time to scrape the coordinator REST API |

### `nixl` — NIXL native telemetry (`:19090`)

NIXL's Prometheus-cpp exporter. Metrics are tagged with `agent_name`
(the NIXL agent UUID) and `hostname`. Only one worker wins the port
under LMCache MP mode.

| Metric | Type | Description |
| --- | --- | --- |
| `agent_tx_bytes_total` | counter | Bytes sent by the NIXL agent (L2 NVMe writes) |
| `agent_rx_bytes_total` | counter | Bytes received by the NIXL agent (L2 NVMe reads) |
| `agent_tx_requests_num_total` | counter | Transfer requests sent |
| `agent_rx_requests_num_total` | counter | Transfer requests received |
| `agent_xfer_time_total` | counter | Cumulative transfer time start-to-complete (µs) |
| `agent_xfer_post_time_total` | counter | Cumulative time from start to posting to back-end (µs) |
| `agent_memory_registered_total` | counter | Cumulative bytes registered with NIXL |
| `agent_memory_deregistered_total` | counter | Cumulative bytes deregistered |
| `agent_memory_registered` | gauge | Currently registered memory (bytes) |
| `agent_memory_deregistered` | gauge | Currently deregistered memory (bytes) |

> **Derived metric (Grafana):** average transfer latency =
> `rate(agent_xfer_time_total[$__rate_interval]) / rate(agent_tx_requests_num_total[$__rate_interval])`

### `node_exporter` — host CPU/mem/disk/network (`:9100`)

Key metric families (full list at
[prometheus.io/docs/guides/node-exporter](https://prometheus.io/docs/guides/node-exporter/)):

| Metric family | Description |
| --- | --- |
| `node_cpu_seconds_total` | CPU time by mode (`user`, `system`, `idle`, `iowait`, …) |
| `node_load1/5/15` | System load averages |
| `node_memory_*` | Memory stats (MemTotal, MemFree, Buffers, Cached, …) |
| `node_disk_read/written_bytes_total` | Disk throughput |
| `node_disk_reads/writes_completed_total` | Disk IOPS |
| `node_disk_read/write_time_seconds_total` | Disk service time |
| `node_nvme_namespace_size_bytes` | NVMe namespace capacity |
| `node_nvme_namespace_used_bytes` | NVMe namespace usage |
| `node_filesystem_avail_bytes` | Filesystem free space |
| `node_pressure_io_stalled_seconds_total` | IO pressure (PSI stall) |
| `node_pressure_memory_stalled_seconds_total` | Memory pressure (PSI stall) |
| `node_infiniband_port_data_received_bytes_total` | IB/RoCE port RX bytes |
| `node_infiniband_port_data_transmitted_bytes_total` | IB/RoCE port TX bytes |
| `node_infiniband_*` | Full IB port counters (errors, link state, …) |

### `nvme_exporter` — NVMe SMART (`:9998`)

| Metric | Description |
| --- | --- |
| `nvme_temperature` | Drive temperature (°C) |
| `nvme_available_spare` | Available spare capacity (%) |
| `nvme_spare_thresh` | Spare threshold (%) |
| `nvme_percent_used` | Percentage of rated life used |
| `nvme_data_units_read` | Data units read (512 KiB blocks) |
| `nvme_data_units_written` | Data units written (512 KiB blocks) |
| `nvme_host_read_commands` | Host read commands |
| `nvme_host_write_commands` | Host write commands |
| `nvme_power_on_hours` | Power-on hours |
| `nvme_power_cycles` | Power cycle count |
| `nvme_unsafe_shutdowns` | Unsafe shutdown count |
| `nvme_media_errors` | Media and data integrity errors |
| `nvme_num_err_log_entries` | Error log entries |
| `nvme_controller_busy_time` | Controller busy time (minutes) |
| `nvme_critical_warning` | Critical warning bitmask |
| `nvme_critical_comp_time` | Critical composite temperature time |
| `nvme_warning_temp_time` | Warning composite temperature time |
| `nvme_used_bytes` | Used capacity (bytes) |
| `nvme_physical_size` | Physical capacity (bytes) |

### `rdma_exporter` — RDMA/RoCE sysfs (`:9879`)

Key throughput and error metrics. Full list of 80+ counters at
[sysfs hw_counters](https://www.kernel.org/doc/html/latest/infiniband/sysfs.html).

| Metric | Description |
| --- | --- |
| `rdma_port_rcv_data_total` | Data octets received (÷4, double-word units) |
| `rdma_port_xmit_data_total` | Data octets transmitted (÷4) |
| `rdma_port_rcv_packets_total` | Packets received |
| `rdma_port_xmit_packets_total` | Packets transmitted |
| `rdma_port_rcv_errors_total` | Receive errors |
| `rdma_port_xmit_discards_total` | Transmit discards |
| `rdma_rx_bytes_total` | RX bytes (hw_counters) |
| `rdma_tx_bytes_total` | TX bytes (hw_counters) |
| `rdma_rx_roce_good_bytes_total` | RoCE good RX bytes |
| `rdma_tx_roce_only_bytes_total` | RoCE-only TX bytes |
| `rdma_link_downed_total` | Link down events |
| `rdma_link_error_recovery_total` | Link error recovery events |
| `rdma_local_link_integrity_errors_total` | Local link integrity errors |
| `rdma_symbol_error_total` | Symbol errors |
| `rdma_port_info` | Port metadata (labels: device, port, state, speed) |
| `rdma_active_qps_total` | Active queue pairs |
| `rdma_active_cqs_total` | Active completion queues |

### `vllm` — vLLM serve (`:8000`)

vLLM metrics use the `vllm:` prefix and `model_name` label.

#### Request throughput & queuing

| Metric | Type | Description |
| --- | --- | --- |
| `vllm:request_success_total` | counter | Completed requests by `finished_reason` |
| `vllm:num_requests_running` | gauge | Requests in active model batches |
| `vllm:num_requests_waiting` | gauge | Requests waiting to be scheduled |
| `vllm:num_requests_waiting_by_reason` | gauge | Waiting requests by reason (`capacity`, …) |
| `vllm:num_preemptions_total` | counter | Preemption events |

#### Token accounting

| Metric | Type | Description |
| --- | --- | --- |
| `vllm:prompt_tokens_total` | counter | Prefill tokens processed |
| `vllm:generation_tokens_total` | counter | Generation tokens produced |
| `vllm:prompt_tokens_by_source_total` | counter | Prefill tokens by `source` label: `local_compute`, `local_cache_hit`, `external_kv_transfer` |
| `vllm:prompt_tokens_cached_total` | counter | Cached prompt tokens (local + external) |

#### Latency histograms

| Metric | Type | Description |
| --- | --- | --- |
| `vllm:time_to_first_token_seconds` | histogram | Time to first token (TTFT) |
| `vllm:inter_token_latency_seconds` | histogram | Inter-token latency (ITL / TPOT) |
| `vllm:e2e_request_latency_seconds` | histogram | End-to-end request latency |
| `vllm:request_prefill_time_seconds` | histogram | Time in prefill phase |
| `vllm:request_decode_time_seconds` | histogram | Time in decode phase |
| `vllm:request_inference_time_seconds` | histogram | Total inference time |
| `vllm:request_queue_time_seconds` | histogram | Time waiting in queue |
| `vllm:request_time_per_output_token_seconds` | histogram | Per-output-token latency |

#### Sequence length distributions

| Metric | Type | Description |
| --- | --- | --- |
| `vllm:request_prompt_tokens` | histogram | Input sequence length (ISL) distribution |
| `vllm:request_generation_tokens` | histogram | Output sequence length (OSL) distribution |
| `vllm:request_max_num_generation_tokens` | histogram | Max generation tokens per request |
| `vllm:iteration_tokens_total` | histogram | Tokens per engine step |

#### KV cache & prefix cache

| Metric | Type | Description |
| --- | --- | --- |
| `vllm:kv_cache_usage_perc` | gauge | KV cache usage (0.0–1.0) |
| `vllm:prefix_cache_hits_total` | counter | GPU-side prefix cache hits (tokens) |
| `vllm:prefix_cache_queries_total` | counter | GPU-side prefix cache queries (tokens) |
| `vllm:external_prefix_cache_hits_total` | counter | LMCache external KV hits (tokens) |
| `vllm:external_prefix_cache_queries_total` | counter | LMCache external KV queries (tokens) |

#### MFU / memory bandwidth

| Metric | Type | Description |
| --- | --- | --- |
| `vllm:estimated_flops_per_gpu_total` | counter | Estimated FLOPs per GPU (for MFU calculation) |
| `vllm:estimated_read_bytes_per_gpu_total` | counter | Estimated memory read bytes per GPU |
| `vllm:estimated_write_bytes_per_gpu_total` | counter | Estimated memory write bytes per GPU |
| `vllm:model_weights_memory_bytes` | gauge | GPU memory used by model weights (static after load) |

#### Misc

| Metric | Type | Description |
| --- | --- | --- |
| `vllm:cache_config_info` | gauge | LLMEngine CacheConfig info (labels) |
| `vllm:engine_sleep_state` | gauge | Engine sleep state (0 = sleeping, 1 = awake) |
| `vllm:mm_cache_hits_total` | counter | Multi-modal cache hits |
| `vllm:mm_cache_queries_total` | counter | Multi-modal cache queries |
| `http_requests_total` | counter | HTTP requests by method/status/handler (FastAPI) |
| `http_request_duration_seconds` | histogram | HTTP latency by handler |

---

## Exporter modes (`AIC_EXPORTERS`)

| Value | What starts | Use case |
| --- | --- | --- |
| `0` | Nothing — Prometheus only | Rely on host-installed exporters |
| `safe` | amdgpu-exporter, rdma-exporter, hsa-snoop (container PID) | **SPUR default** |
| `1` | Full fleet: node-exporter, amdgpu-exporter, hsa-snoop (host PID), fabric exporters | Non-SPUR bare nodes |

| Profile | Services |
| --- | --- |
| `exporters` | node-exporter, amdgpu-exporter, hsa-snoop (`pid:host`) |
| `exporters-safe` | amdgpu-exporter, rdma-exporter, hsa-snoop (`pid:container`) |
| `exporters-fabric` | nvme-exporter, rdma-exporter |

```bash
# Exporters already on host (Ansible)
make monitoring-up AIC_METRICS_DIR=/mnt/lmcache-nfs/metrics

# Safe mode (SPUR / authz-restricted)
make monitoring-up AIC_METRICS_DIR=/mnt/lmcache-nfs/metrics AIC_EXPORTERS=safe

# Full fleet (bare node)
make monitoring-up AIC_METRICS_DIR=/mnt/lmcache-nfs/metrics AIC_EXPORTERS=1

make monitoring-down     # stop (TSDB retained)
```

---

## hsa-snoop (`:9488`)

[sbates130272/hsa-snoop](https://github.com/sbates130272/hsa-snoop) is compiled
into the `rocm-aic` image (`-DHSA_SNOOP_PROMETHEUS=ON`). It exports HSA AQL
dispatch metrics and AIS P2P storage counters.

**PID namespace modes:**

| Mode | `AIC_HSA_SNOOP_PID_MODE` | When to use |
| --- | --- | --- |
| Container PID | `container:aic-lmcache` | **SPUR default** — vLLM uses `pid:service:lmcache` so lmcache's namespace contains both processes |
| Host PID | `host` | Non-SPUR bare nodes where `--pid=host` is permitted |

> **SPUR authz note:** the SPUR cluster's Docker authz plugin blocks `--pid=host`.
> Use `AIC_EXPORTERS=safe` to launch hsa-snoop with `pid:container`.

hsa-snoop holds tracefs exclusively on the host, so it is always scraped via
`host-gateway:9488` rather than a container name.

---

## fabric exporters (nvme\_exporter / rdma\_exporter)

Normally host services (Ansible roles). Container images are built from the
same upstream release binaries when host exporters are absent:

```bash
# Build both fabric-exporter images
make monitoring-build-exporters

# Run via exporters-fabric compose profile
AIC_METRICS_DIR=/mnt/lmcache-nfs/metrics \
  docker compose -f docker/docker-compose.yml \
    --profile exporters --profile exporters-fabric up -d
```

> **Note:** these images pull `debian:12-slim` from Docker Hub at build time;
> the build node needs registry egress. Skip with `AIC_BUILD_EXPORTERS=0`.
> `make dist-build` treats failure as a non-fatal warning.

---

## Container log streaming

During each cliff arm, `docker logs -f --timestamps` is streamed into:

```text
logs/<job-id>/lmcache/lmcache-stream.log
logs/<job-id>/vllm/vllm-stream.log
```

These complement the Prometheus TSDB for fine-grained timing — per-token
store times, NIXL errors, and EngineCore stack traces are visible in real
time rather than only at teardown.

---

## NFS caveat

Prometheus' TSDB uses `mmap` + POSIX file locks, which NFS handles poorly.
Keep to a single writer — fine for lab/demo capture, not a durable production
store.
