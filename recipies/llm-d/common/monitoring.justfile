# Common monitoring utilities for all deployments
# Import this file in deployment-specific justfiles with: import? "monitoring.justfile"

# Default monitoring namespace used by llm-d
MONITORING_NAMESPACE := env_var_or_default('MONITORING_NAMESPACE', 'llm-d-monitoring')

# Show Grafana login information
grafana-info:
    @echo "Access Grafana at: http://localhost:3000 (after port-forward)"

# Internal: Start Prometheus port-forward (shared across deployments)
_port-forward-prometheus:
    #!/usr/bin/env bash
    set -euo pipefail
    # Kill existing port-forward if running
    pkill -f "port-forward.*prometheus.*9090" 2>/dev/null || true
    sleep 1
    # Start new port-forward
    kubectl port-forward -n {{MONITORING_NAMESPACE}} svc/llmd-kube-prometheus-stack-prometheus 9090:9090 > /dev/null 2>&1 &
    echo "  ✅ Prometheus port-forward started (PID: $!)"

# Internal: Start Grafana port-forward (shared across deployments)
_port-forward-grafana:
    #!/usr/bin/env bash
    set -euo pipefail
    # Kill existing port-forward if running
    pkill -f "port-forward.*grafana.*3000" 2>/dev/null || true
    sleep 1
    # Start new port-forward
    kubectl port-forward -n {{MONITORING_NAMESPACE}} svc/llmd-grafana 3000:80 > /dev/null 2>&1 &
    echo "  ✅ Grafana port-forward started (PID: $!)"

# Stop monitoring port-forwards
stop-monitoring-port-forwards:
    @echo "Stopping monitoring port-forwards..."
    -pkill -f "port-forward.*prometheus.*9090"
    -pkill -f "port-forward.*grafana.*3000"
    @echo "✅ Monitoring port-forwards stopped"

# Parse vLLM configuration from pod logs (single pod)
_parse-vllm-pod NAMESPACE POD_NAME:
    @python3 parse-vllm-config.py -n {{NAMESPACE}} -p {{POD_NAME}}

# Parse vLLM configuration from all pods (with deduplication)
_parse-vllm-all NAMESPACE LABEL_SELECTOR="llm-d.ai/inference-serving=true" DEDUPE="true":
    #!/usr/bin/env bash
    if [ "{{DEDUPE}}" = "true" ]; then
        python3 parse-vllm-config.py -n {{NAMESPACE}} -l {{LABEL_SELECTOR}} --deduplicate
    else
        python3 parse-vllm-config.py -n {{NAMESPACE}} -l {{LABEL_SELECTOR}}
    fi

# Parse SGLang configuration from pod logs (single pod)
_parse-sglang-pod NAMESPACE POD_NAME:
    @python3 parse-sglang-config.py -n {{NAMESPACE}} -p {{POD_NAME}}

# Parse SGLang configuration from all pods (with deduplication)
_parse-sglang-all NAMESPACE LABEL_SELECTOR="llm-d.ai/inference-serving=true" DEDUPE="true":
    #!/usr/bin/env bash
    if [ "{{DEDUPE}}" = "true" ]; then
        python3 parse-sglang-config.py -n {{NAMESPACE}} -l {{LABEL_SELECTOR}} --deduplicate
    else
        python3 parse-sglang-config.py -n {{NAMESPACE}} -l {{LABEL_SELECTOR}}
    fi

# Load Grafana dashboards (llm-d + rocm-aic custom)
_load-dashboards:
    @../monitoring/load-dashboards.sh {{MONITORING_NAMESPACE}}
