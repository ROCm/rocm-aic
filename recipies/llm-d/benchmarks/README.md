# LLM-D Benchmarking Infrastructure

Automated benchmarking sweep infrastructure for exploring vLLM deployment parameter spaces.

## Quick Start

```bash
# Set required environment variable
export HF_TOKEN=your_huggingface_token_here

# List available sweep configurations
just list-sweeps

# Preview what a sweep will do (dry-run)
just dry-run test-small-offloading.yaml

# Run the example sweep
just sweep test-small-offloading.yaml

# Currently executing namespaces
just current test-small-offloading.yaml

# Status of a currently executing namespace
just status <namespace executing>

# View results
just list-results
just show-summary <sweep-dir-name>
```

## Overview

This infrastructure automates:
1. **Namespace Creation**: Each run gets a unique namespace (user-prefixed)
2. **Secret Injection**: HuggingFace token injected into namespace
3. **Deployment**: Generate and deploy parameterized vLLM configurations
4. **Load Generation**: Run multi-turn conversation benchmarks
5. **Results Collection**: Save pod logs and benchmark outputs
6. **Teardown**: Delete namespace and clean up all resources

## Prerequisites

- Kubernetes cluster access with `kubectl` configured
- **Required**: `HF_TOKEN` environment variable set with your HuggingFace token
- Python 3.7+ with `pyyaml` installed

## Directory Structure

```
benchmarks/
├── sweep-configs/          # Sweep configuration files
├── templates/              # Deployment templates
├── load-generators/        # Load generation tools
│   └── multi-turn-benchmark/
├── scripts/                # Orchestration scripts
└── results/                # Benchmark results (gitignored)
    └── sweeps/
        └── TEST_NAME-YYYY-MM-DD/
            ├── run-001/
            │   ├── config.yaml
            │   ├── manifests/
            │   ├── benchmark_runner_output.txt
            │   ├── benchmark_output.txt
            │   └── snapshots/
            ├── run-002/
            └── summary.json
```

## Configuration Format

Sweep configurations use a simplified approach where parameter names match vLLM arguments verbatim:

```yaml
name: "my-sweep"
description: "Description of the sweep"
deployment: "tiered-prefix-cache"
# Note: Namespaces are auto-generated per run (no need to specify)

parameters:
  model:
    type: fixed
    value: "Qwen/Qwen3-32B"

  tensor_parallel_size:
    type: fixed
    value: 2

  # Optional: Specify number of replicas for the decode deployment (default: 1)
  replicas:
    type: fixed
    value: 1

  # Optional: Specify the container image for the decode deployment
  # If not specified, uses the default image from the template
  image:
    type: fixed
    value: "ghcr.io/vcave/vllm:rocm_721-vllm_0190-lmcache_20260514_92fe433a"

  # Optional: Specify CPU resource limit/request (default: varies by template)
  cpu:
    type: fixed
    value: "32"

  # Optional: Specify memory resource limit/request (default: varies by template)
  memory:
    type: fixed
    value: "100Gi"

  # vLLM arguments with automatic conversion to CLI flags
  vllm_args:
    type: combinations
    args:
      max_num_seq:
        values: [512, 1024]

      gpu_memory_utilization:
        values: [0.90, 0.95]

      # KV connector shortcuts
      kv_connector:
        values:
          - null  # No caching
          - type: "offloading"
            cpu_bytes: 107374182400  # 100GB

load_generation:
  tool: "multi-turn-benchmark"
  image: "vllm/vllm-openai:latest"  # Docker image with benchmark
  workload_file: "agent_multi_turn.json"  # Workload to mount
  benchmark_args:
    num_clients: 100
    max_active_conversations: 100
    # Note: input_file auto-added as /workload/{workload_file}
```

### KV Connector Shortcuts

Simplified configuration for common KV cache setups:

**CPU Offloading:**
```yaml
kv_connector:
  type: "offloading"
  cpu_bytes: 107374182400  # 100GB in bytes
  role: "kv_both"           # Optional, defaults to kv_both
```

**LMCache:**
```yaml
kv_connector:
  type: "lmcache"
  role: "kv_both"
  config_file: "/etc/lmcache/lmcache-cpu.yaml"
```

**Custom (Raw JSON):**
```yaml
kv_connector:
  raw_json: '{"kv_connector":"CustomConnector","kv_role":"kv_both"}'
```

### Environment Variables

The sweep orchestrator supports injecting environment variables into Kubernetes Pod containers. This feature enables configuration of runtime behavior, debugging settings, and integration with external services.

#### Global Environment Variables

Define environment variables at the top level to apply them to all sweep runs:

```yaml
name: "my-sweep"
deployment: "inference-scheduling-vllm"

# Global environment variables (applied to all runs)
env_vars:
  LOG_LEVEL: "INFO"
  ENABLE_METRICS: "true"
  OTEL_SERVICE_NAME: "vllm-benchmark"

parameters:
  model:
    type: fixed
    value: "facebook/opt-125m"
  # ... rest of config
```

#### Per-Combination Environment Variables

Override or add environment variables for specific parameter combinations:

```yaml
parameters:
  tensor_parallel_size:
    type: categorical
    values: [1, 2, 4]
    # Each entry corresponds to a value in the values list
    env_vars:
      - # For TP=1
        TP_SIZE: "1"
        CACHE_SIZE: "10GB"
      - # For TP=2
        TP_SIZE: "2"
        CACHE_SIZE: "20GB"
      - # For TP=4
        TP_SIZE: "4"
        CACHE_SIZE: "40GB"
        ENABLE_TRACING: "true"  # Only for TP=4
```

**Override Semantics**: Combination-level `env_vars` override global `env_vars` for the same keys.

#### Host Environment Variable Substitution

Use `${VAR}` syntax to substitute values from the host environment at sweep configuration load time:

```yaml
env_vars:
  # With default value (if HOST_VAR not set, use default)
  API_BASE_URL: "${BASE_URL:-http://localhost:8000}"

  # Without default (remains as ${VAR} if not set)
  CUSTOM_CONFIG: "${MY_CUSTOM_CONFIG}"

  # Works in any config value, not just env_vars
model:
  type: fixed
  value: "${MODEL_NAME:-facebook/opt-125m}"
```

**Syntax**:
- `${VAR}` - Substitute from host environment (remains unchanged if not set)
- `${VAR:-default}` - Substitute with default value if VAR is not set

#### Example

See `sweep-configs/example-env-vars.yaml` for a complete example demonstrating:
- Global environment variables
- Per-combination environment variables
- Host environment variable substitution
- Override semantics

#### Runtime YAML Defaults

The runner loads checked-in defaults from `runtime-defaults.yaml`. For
host-specific settings that should apply across many sweeps, put overrides in
`runtime.yaml`, or pass a different override file:

```bash
python3 scripts/run-sweep.py my-sweep --runtime-config /path/to/runtime.yaml
```

Runtime YAML can set the sweep results directory, default pod `env_vars`, and
Hugging Face token-file location. Sweep config `env_vars` still override
runtime `env_vars`, and host environment variables still override host runtime
settings such as `sweep.results_dir`.

### Deployment Configuration

#### Replicas, Image, CPU, and Memory

You can specify the number of replicas, container image, and resource limits for the decode deployment:

```yaml
parameters:
  # Number of replicas for the decode deployment
  replicas:
    type: fixed
    value: 2

  # Container image for the decode deployment
  image:
    type: fixed
    value: "ghcr.io/vcave/vllm:your-custom-image"

  # CPU resource limit/request (same value for both)
  cpu:
    type: fixed
    value: "32"

  # Memory resource limit/request (same value for both)
  memory:
    type: fixed
    value: "100Gi"
```

**Use cases:**
- **Multiple replicas**: Scale out for higher throughput or availability
- **Custom images**: Test different vLLM versions, custom builds, or optimizations
- **Resource tuning**: Find optimal CPU and memory allocations for your workload
- **Sweeping configurations**: Compare performance across different resource configurations

All parameters are optional and can be swept like any other parameter:

```yaml
parameters:
  replicas:
    type: categorical
    values: [1, 2, 4]

  image:
    type: categorical
    values:
      - "ghcr.io/vcave/vllm:rocm_721-vllm_0190-lmcache_20260514_92fe433a"
      - "ghcr.io/vcave/vllm:rocm_721-vllm_20260501-lmcache"

  cpu:
    type: categorical
    values: ["16", "32", "64"]

  memory:
    type: categorical
    values: ["50Gi", "100Gi", "200Gi"]
```

**Examples:**
- `sweep-configs/example-replicas-and-image.yaml` - Basic usage
- `sweep-configs/example-replicas-image-sweep.yaml` - Sweeping replicas and images
- `sweep-configs/example-resource-sweep.yaml` - Sweeping CPU and memory resources

## Running Sweeps Quick start

This section and implementation are still under development.

Run the lmcache TTFT latency benchmark sweep:
```bash
just sweep sweep-configs/ttft-latency/bench-ttft-lmcache-cpu-only.yaml
```


Upon completion, locate the output folder and aggregate results into json file:
```bash
mkdir -p results/ttft-latency
just results-aggregate ttft-lmcache-cpu-only_2026-04-21 -o results/ttft-latency/lmcache_cpu_aggregated_results.json
```

Generate png and json description in current folder out of a plot configuration file:
```bash
pip install pandas typing-extensions matplotlib seaborn pyyaml
export PYTHONPATH=$PWD/scripts
python -m plots.plot_config sweep-configs/ttft-latency/plot-config-ttft-lmcache.yaml
```
Note the input result json file are declared in the plot configuration file and must match the path where the aggregate results file have been output.


## Running Sweeps

### Dry Run (Preview)

Before running a sweep, preview what configurations will be generated:

```bash
# Preview configurations
just dry-run test-small-offloading

# Or use the script directly
python3 scripts/run-sweep.py sweep-configs/test-small-offloading --dry-run
```

This will show:
- Total number of configurations
- Each configuration's parameters
- Generated vLLM commands
- Load generation settings

**Example output:**
```
======================================================================
DRY RUN: example-cache-sweep
======================================================================
Description: Compare different KV cache strategies
Total configurations: 3

Configuration 1/3
----------------------------------------------------------------------
  model: Qwen/Qwen3-32B
  vllm_args:
    kv_connector: null  # No caching
  Generated vLLM command:
    vllm serve Qwen/Qwen3-32B --max-num-seq 512 ...
...
======================================================================
SUMMARY: 3 configurations would be executed
======================================================================
```

### Basic Usage

```bash
# Run a sweep
just sweep test-small-offloading.yaml

# The sweep will:
# 1. Generate all parameter combinations
# 2. For each combination:
#    - Create unique namespace (format: <user>-llmd-bench-<timestamp>-<run-id>)
#    - Inject HF_TOKEN as Kubernetes secret (hf-secret)
#    - Render deployment manifests
#    - Deploy to Kubernetes in the unique namespace
#    - Wait for readiness
#    - Run load generation
#    - Collect pod logs
#    - Delete namespace (full teardown)
# 3. Save results and summary
```

### Namespace Isolation

Each configuration run gets its own unique Kubernetes namespace:

**Format**: `<user-id>-llmd-bench-<timestamp>-<run-id>`

**Example**: `vinccave-llmd-bench-2026-03-30_13-21-40-001`

**Benefits**:
- Complete isolation between runs
- No resource conflicts
- Clean slate for each configuration
- Easy cleanup (delete namespace)
- Traceable to user and sweep

**Note**: The user ID is automatically detected from your environment (`$USER` or `whoami`).
```

### Viewing Results

```bash
# List all sweep runs
just list-results

# Show summary of a specific sweep
just show-summary 2024-03-30_14-30-00_example-cache-sweep

# View configuration for a specific run
just show-run-config 2024-03-30_14-30-00_example-cache-sweep 001

# View logs from a specific run
just show-logs 2024-03-30_14-30-00_example-cache-sweep 001
```

## Results Structure

Each sweep run saves:

1. **config.yaml**: The parameter configuration for this run
2. **manifests/**: Rendered Kubernetes manifests
3. **benchmark_output.txt**: Benchmark stdout
4. **benchmark_error.txt**: Benchmark stderr
5. **snapshots/**: Namespace diagnostics snapshot
   - `<pod-name>.log`: Complete logs from each pod
   - `<pod-name>-describe.txt`: Pod describe output
   - `namespace-events.yaml`: Kubernetes events
   - `metadata.json`: Snapshot collection metadata

The sweep summary (`summary.json`) contains:
- List of all runs with their parameters
- Success/failure status for each run
- Benchmark result file paths

## Advanced Configuration

### Sweeping Multiple Parameters

```yaml
vllm_args:
  type: combinations
  args:
    max_num_seq:
      values: [512, 1024, 2048]

    gpu_memory_utilization:
      values: [0.85, 0.90, 0.95]

    kv_connector:
      values:
        - null
        - type: "offloading"
          cpu_bytes: 53687091200
        - type: "offloading"
          cpu_bytes: 107374182400

# This creates: 3 × 3 × 3 = 27 configurations
```

### Sweeping Load Parameters

```yaml
load_generation:
  tool: "multi-turn-benchmark"
  image: "vllm/vllm-openai:latest"
  workload_file: "agent_multi_turn.json"
  benchmark_args:
    num_clients: 300
    max_active_conversations: 300
    verbose: true
```

## Monitoring Progress

This feature allows you to interact with currently executing namespaces during benchmark sweeps. You can view logs, check status, and inspect configurations without stopping the sweep.

### Key Features

1. **`just current <config>`** - List all active namespaces for a sweep
2. **Namespace-aware targets** - Interact with specific namespaces:
   - `just logs <namespace>` - View logs
   - `just logs-follow <namespace>` - Follow logs in real-time
   - `just show-config <namespace>` - Display vLLM configuration
   - `just status <namespace>` - Check deployment status

These targets are essentially shortcuts for resolving the type of deployment and using the associated justfile that implement these targets natively.

### Load Generator Execution

The benchmark runs as a Kubernetes pod:

1. **ConfigMap Creation**: Workload file embedded in ConfigMap
2. **Pod Creation**: Benchmark pod created with ConfigMap mounted at `/workload/`
3. **Single kubectl apply**: Both resources created atomically
4. **Log Capture**: Pod stdout/stderr streamed to files
5. **Cleanup**: Pod deleted after completion (ConfigMap auto-deleted with namespace)

**Pod Details:**
- Image: Configurable via `image` field
- Workload: Mounted at `/workload/{workload_file}`
- Service URL: Auto-generated from namespace
- Timeout: 3600s (1 hour)
- Logs: Saved to `benchmark_output.txt` and `pod_description.txt`

## Design Principles

1. **Simplified Arguments**: Parameter names match vLLM/benchmark CLI arguments exactly
2. **Minimal Abstraction**: No custom layers - direct pass-through to tools
3. **KV Connector Shortcuts**: Easy configuration for common caching strategies
4. **Results First**: Save all outputs for later analysis

## Current Limitations

This is a minimal implementation focused on sweep execution:

- **No metrics collection**: Prometheus integration not yet implemented
- **No results analysis**: Analysis tools to be added later
- **Simulated benchmark**: Load generator is currently a placeholder
- **No retry logic**: Failed runs are not automatically retried

## Next Steps

Planned enhancements:
1. Integrate actual multi-turn benchmark execution
2. Add Prometheus metrics collection
3. Implement results analysis and visualization
4. Add retry logic for transient failures
5. Support parallel sweep execution

## See Also

- [BENCHMARKING_PROPOSAL.md](../BENCHMARKING_PROPOSAL.md) - Full design proposal
- [test-small-offloading](sweep-configs/test-small-offloading) - Example configuration
