#!/bin/bash
# Long document QA runner
# Executes long_doc_qa.py in a Kubernetes pod

set -euo pipefail

# Default values
IMAGE=""
NAMESPACE=""
OUTPUT_DIR=""
RESULTS_DIR=""
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
    --results-dir)
      RESULTS_DIR="$2"
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
      echo "Usage: $0 --image IMAGE --namespace NS --output-dir DIR --results-dir DIR [--run-label LABEL] [--dry-run] -- <benchmark-args>"
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

# If results-dir not specified, default to /tmp/benchmark-results
if [ -z "$RESULTS_DIR" ]; then
  RESULTS_DIR="/tmp/benchmark-results/${NAMESPACE}"
  echo "INFO: --results-dir not specified, using default: $RESULTS_DIR"
fi

# Generate pod name
POD_NAME="long-doc-qa-${RUN_LABEL}-$(date +%s)"

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
    tool: long-doc-qa
spec:
  restartPolicy: Never
  containers:
  - name: benchmark
    image: ${IMAGE}
    command: ["/bin/sh", "-c"]
    args:
    - |
      # Run benchmark (long_doc_qa.py outputs to current directory)
      # TODO fix this; racy; hard-coded
      rm -Rf /mnt/rocm-icms-cache/vinccave/*
      python /utils/long_doc_qa.py ${ARGS_STR}
      rm -f ${RESULTS_DIR}/*
      cp -v *.csv /results 2>/dev/null || echo "No .csv files to copy"
      cp -v *.png /results 2>/dev/null || echo "No .png files to copy"
    volumeMounts:
    - name: results
      mountPath: /results
    - name: utils
      mountPath: /utils
  volumes:
  - name: results
    hostPath:
      path: ${RESULTS_DIR}
      type: DirectoryOrCreate
  - name: utils
    hostPath:
      path: /mnt/rocm-icms-cache/utils
      type: DirectoryOrCreate

EOF

# Dry-run mode
if [ "$DRY_RUN" = true ]; then
  echo "=========================================="
  echo "DRY RUN: Long Document QA"
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
  echo "Running long_doc_qa..."
  echo "Warmup round completed"
  echo "Query round completed"
  echo "Results saved to CSV files"
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

# Create output directory structure
mkdir -p "$OUTPUT_DIR/$RUN_LABEL"
LOG_FILE="$OUTPUT_DIR/benchmark_output_${RUN_LABEL}.log"

# Wait for pod completion
echo "Waiting for benchmark to complete (timeout: 10800s)..."
kubectl wait --for=condition=Ready=false "pod/$POD_NAME" -n "$NAMESPACE" --timeout=10800s 2>/dev/null || true

# Get logs
echo "Retrieving logs..."
kubectl logs "$POD_NAME" -n "$NAMESPACE" > "$LOG_FILE" 2>&1 || true

# Copy result files from pod's /results volume to host OUTPUT_DIR
echo "Copying result files from shared results folder to host..."
cp ${RESULTS_DIR}/* ${OUTPUT_DIR}

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
echo "Results saved to: $OUTPUT_DIR/$RUN_LABEL/"

exit "$EXIT_CODE"
