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
            │   └── logs/
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
5. **logs/**: vLLM pod logs
   - `<pod-name>.log`: Complete logs from each vLLM pod

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
