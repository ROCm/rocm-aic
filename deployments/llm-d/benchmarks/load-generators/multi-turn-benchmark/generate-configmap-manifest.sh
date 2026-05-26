#!/bin/bash
# Multi-turn Benchmark ConfigMap Manifest Generator
# Generates Kubernetes ConfigMap YAML for workload data

# Expected environment variables:
# - NAMESPACE: Kubernetes namespace
# - WORKLOAD_FILE: Path to the workload file (must be provided by caller)

if [ -z "$WORKLOAD_FILE" ]; then
  echo "ERROR: WORKLOAD_FILE environment variable is required" >&2
  exit 1
fi

if [ ! -f "$WORKLOAD_FILE" ]; then
  echo "ERROR: Workload file not found: $WORKLOAD_FILE" >&2
  exit 1
fi

WORKLOAD_BASENAME=$(basename "$WORKLOAD_FILE")
WORKLOAD_CONTENT=$(cat "$WORKLOAD_FILE" | sed 's/^/    /')

cat << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: benchmark-workload
  namespace: ${NAMESPACE}
data:
  ${WORKLOAD_BASENAME}: |
${WORKLOAD_CONTENT}
EOF
