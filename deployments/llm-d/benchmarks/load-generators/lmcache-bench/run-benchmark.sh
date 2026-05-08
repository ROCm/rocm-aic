#!/bin/bash
# LMCache bench runner
# Executes lmcache bench engine in a Kubernetes pod

set -euo pipefail

# Default values
IMAGE=""
NAMESPACE=""
OUTPUT_DIR=""
RUN_LABEL="run1"
DRY_RUN=false
BENCHMARK_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --namespace)
      NAMESPACE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --run-label)
      RUN_LABEL="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --)
      shift
      BENCHMARK_ARGS=("$@")
      break
      ;;
    *)
      echo "ERROR: Unknown argument: $1"
      echo "Usage: $0 --image IMAGE --namespace NS --output-dir DIR [--run-label LABEL] [--dry-run] -- <benchmark-args>"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [ -z "$IMAGE" ]; then
  echo "ERROR: --image is required"
  exit 1
fi

if [ -z "$NAMESPACE" ]; then
  echo "ERROR: --namespace is required"
  exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
  echo "ERROR: --output-dir is required"
  exit 1
fi

# Generate pod name (use timestamp + run_label for uniqueness)
POD_NAME="lmcache-bench-${RUN_LABEL}-$(date +%s)"

# Build benchmark args string for YAML
ARGS_STR=""
for arg in "${BENCHMARK_ARGS[@]}"; do
  ARGS_STR+="$arg "
done

# Generate Pod YAML
POD_YAML=$(mktemp)

cat > "$POD_YAML" << EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${POD_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: llm-d-benchmark
    tool: lmcache-bench
spec:
  restartPolicy: Never
  containers:
  - name: benchmark
    image: ${IMAGE}
    command: ["/bin/sh", "-c"]
    args:
    - |
      # Run lmcache bench engine
      lmcache bench engine ${ARGS_STR}
    volumeMounts:
    - name: bench-config-volume
      mountPath: /opt/bench-configs
      readOnly: true
  volumes:
  - name: bench-config-volume
    configMap:
      name: bench-config-map
EOF

# Dry-run mode
if [ "$DRY_RUN" = true ]; then
  echo "=========================================="
  echo "DRY RUN: LMCache Bench"
  echo "=========================================="
  echo ""
  echo "Would execute:"
  echo "  kubectl apply -f <pod-yaml> -n $NAMESPACE"
  echo ""
  echo "Generated YAML:"
  echo "----------------------------------------"
  cat "$POD_YAML"
  echo "----------------------------------------"
  echo ""
  echo "Mock Output:"
  echo "pod/$POD_NAME created"
  echo ""
  echo "Benchmark execution (simulated):"
  echo "Running lmcache bench engine..."
  echo "Completed 100 queries"
  echo "Mean TTFT: 125ms"
  echo "Cache hit rate: 75%"
  echo "Throughput: 45 req/s"
  echo "=========================================="
  rm -f "$POD_YAML"
  exit 0
fi

# Execute benchmark
echo "Creating benchmark pod in namespace $NAMESPACE..."

if ! kubectl apply -f "$POD_YAML" -n "$NAMESPACE"; then
  echo "ERROR: Failed to create pod"
  rm -f "$POD_YAML"
  exit 1
fi

echo "Pod created successfully"
rm -f "$POD_YAML"

# Wait for pod to start
echo "Waiting for pod to be ready..."
if ! kubectl wait --for=condition=Ready "pod/$POD_NAME" -n "$NAMESPACE" --timeout=300s 2>/dev/null; then
  echo "Warning: Pod did not become ready, but continuing to capture logs..."
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/benchmark_output_${RUN_LABEL}.log"

# Wait for pod completion
echo "Waiting for benchmark to complete (timeout: 1200s)..."
kubectl wait --for=condition=Ready=false "pod/$POD_NAME" -n "$NAMESPACE" --timeout=1200s 2>/dev/null || true

# Get logs
echo "Retrieving logs..."
kubectl logs "$POD_NAME" -n "$NAMESPACE" > "$LOG_FILE" 2>&1 || true

# Save pod description for debugging
POD_DESC_FILE="$OUTPUT_DIR/pod_description_${RUN_LABEL}.txt"
kubectl describe pod "$POD_NAME" -n "$NAMESPACE" > "$POD_DESC_FILE" 2>&1 || true

# Get exit code
EXIT_CODE=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[0].state.terminated.exitCode}' 2>/dev/null || echo "1")

if [ "$EXIT_CODE" != "0" ]; then
  echo "WARNING: Benchmark exited with code $EXIT_CODE"
fi

# Cleanup pod
echo "Cleaning up pod..."
kubectl delete pod "$POD_NAME" -n "$NAMESPACE" --wait=true 2>/dev/null || true

echo "Benchmark execution complete"
echo "Logs saved to: $LOG_FILE"

exit "$EXIT_CODE"
