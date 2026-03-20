# Monitoring Stack Management

This directory manages Grafana dashboard installation for rocm-aic deployments.

## Overview

The monitoring setup uses a **layered approach**:

1. **llm-d default dashboards** - From llm-d submodule (6 dashboards)
2. **rocm-aic custom dashboards** - Custom dashboards in `grafana/dashboards/`

All dashboards are loaded as Kubernetes ConfigMaps with the `grafana_dashboard=1` label, which Grafana automatically discovers and imports.

## Prerequisites

**Monitoring Stack Must Be Deployed First:**

The monitoring stack (Prometheus + Grafana) must be installed in the `llm-d-monitoring` namespace:

```bash
# Deploy monitoring stack from llm-d guide
cd ../../../../submodules/llm-d/docs/monitoring
kubectl apply -k setup/
```

Or follow the llm-d monitoring setup guide.

## Quick Start

### Load All Dashboards

```bash
cd recipies/llm-d/monitoring
just load-dashboards
```

This loads:
- 6 llm-d default dashboards
- All custom dashboards from `grafana/dashboards/*.json`

### Access Grafana

```bash
# Get Grafana credentials
just grafana-access

# From any deployment directory, start port-forward
cd ../tiered-prefix-cache
just port-forward-start

# Open browser: http://localhost:3000
```

## Available Recipes

### Dashboard Management

- `just load-dashboards` - Load all dashboards (llm-d + custom)
- `just load-llm-d-dashboards` - Load only llm-d defaults
- `just load-custom-dashboards` - Load only rocm-aic custom
- `just list-dashboards` - List loaded dashboards
- `just delete-dashboards` - Delete all dashboards
- `just reload-dashboards` - Delete and reload all dashboards

### Monitoring Info

- `just grafana-access` - Show Grafana URL and credentials
- `just verify-monitoring` - Verify monitoring stack is running

## llm-d Default Dashboards

Located in: `submodules/llm-d/docs/monitoring/grafana/dashboards/`

1. **kv-offload-connector-metrics.json** - KV cache offloading metrics
2. **llm-d-diagnostic-drilldown-dashboard.json** - Diagnostic deep dive
3. **llm-d-failure-saturation-dashboard.json** - Failure analysis and saturation
4. **llm-d-vllm-overview.json** - vLLM overview metrics
5. **llm-performance-kv-cache.json** - KV cache performance
6. **pd-coordinator-metrics.json** - Prefill/Decode coordinator metrics

These are loaded as ConfigMaps with prefix: `llmd-*`

## rocm-aic Custom Dashboards

**Location**: `grafana/dashboards/`

Add your custom Grafana dashboard JSON files here. They will be:
- Loaded as ConfigMaps with prefix: `rocm-aic-*`
- Automatically discovered by Grafana (within 30 seconds)
- Available in Grafana UI under "Dashboards"

### Adding Custom Dashboards

1. **Export from Grafana**:
   - Create/modify dashboard in Grafana UI
   - Share → Export → Save JSON

2. **Save to directory**:
   ```bash
   mv ~/Downloads/my-dashboard.json grafana/dashboards/
   ```

3. **Load into Grafana**:
   ```bash
   just load-dashboards
   ```

4. **Verify**:
   ```bash
   just list-dashboards
   ```

### Example Custom Dashboard

Create `grafana/dashboards/rocm-gpu-utilization.json`:

```json
{
  "dashboard": {
    "title": "ROCm GPU Utilization",
    "panels": [
      {
        "title": "GPU Memory Usage",
        "targets": [
          {
            "expr": "vllm:gpu_cache_usage_perc"
          }
        ]
      }
    ]
  }
}
```

Then load:
```bash
just load-dashboards
```

## Dashboard Organization

**ConfigMap Naming**:
- llm-d: `llmd-{dashboard-name}`
- rocm-aic: `rocm-aic-{dashboard-name}`

**Labels**:
- All dashboards: `grafana_dashboard=1`

**Discovery**:
Grafana sidecar watches for ConfigMaps with `grafana_dashboard=1` label and automatically imports them.

## Troubleshooting

### Dashboards Not Appearing

**Check ConfigMaps**:
```bash
kubectl get configmap -n llm-d-monitoring -l grafana_dashboard=1
```

**Check Grafana Logs**:
```bash
kubectl logs -n llm-d-monitoring -l app.kubernetes.io/name=grafana -c grafana-sc-dashboard
```

**Reload Dashboards**:
```bash
just reload-dashboards
```

### Monitoring Stack Not Running

```bash
# Verify
just verify-monitoring

# Deploy monitoring stack
cd ../../../../submodules/llm-d/docs/monitoring
kubectl apply -k setup/
```

### Dashboard JSON Errors

Validate your JSON:
```bash
jq . grafana/dashboards/my-dashboard.json
```

## Integration with Deployments

### From Deployment Directories

You can load dashboards from any deployment:

```bash
cd recipies/llm-d/tiered-prefix-cache

# Deploy infrastructure
just deploy

# Load dashboards
cd ../monitoring
just load-dashboards

# Start port-forward to access Grafana
cd ../tiered-prefix-cache
just port-forward-start
```

### Automated Setup

Add to deployment setup workflow:

```bash
# In deployment justfile
setup-with-dashboards: setup
    cd ../monitoring && just load-dashboards
```

## References

- [llm-d Monitoring Guide](../../../submodules/llm-d/docs/monitoring/)
- [Grafana Dashboard JSON Schema](https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/create-dashboard/)
- [Grafana Sidecar](https://github.com/grafana/helm-charts/tree/main/charts/grafana#sidecar-for-dashboards)
