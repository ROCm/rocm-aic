#!/bin/bash
# Multi-turn benchmark runner
# Executes vLLM benchmark in a Kubernetes pod with workload mounted via ConfigMap

set -euo pipefail

# Default values
IMAGE=""
NAMESPACE=""
WORKLOAD_FILE=""
OUTPUT_DIR=""
RESULTS_DIR=""
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
    --workload-file)
      WORKLOAD_FILE="$2"
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
      echo "Usage: $0 --image IMAGE --namespace NS --workload-file FILE --output-dir DIR --results-dir DIR [--dry-run] -- <benchmark-args>"
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

if [ -z "$WORKLOAD_FILE" ]; then
  echo "ERROR: --workload-file is required"
  exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
  echo "ERROR: --output-dir is required"
  exit 1
fi

# If results-dir not specified, default to /tmp/benchmark-results
if [ -z "$RESULTS_DIR" ]; then
  RESULTS_DIR="/tmp/benchmark-results"
  echo "INFO: --results-dir not specified, using default: $RESULTS_DIR"
fi

# Validate workload file exists
if [ ! -f "$WORKLOAD_FILE" ]; then
  echo "ERROR: Workload file not found: $WORKLOAD_FILE"
  exit 1
fi

# Extract workload filename
WORKLOAD_BASENAME=$(basename "$WORKLOAD_FILE")

# Generate pod name (use timestamp for uniqueness)
POD_NAME="benchmark-runner-$(date +%s)"

# Generate combined YAML
COMBINED_YAML=$(mktemp)

#TODO parameterize the `agent_multi_turn.json`
cat > "$COMBINED_YAML" << 'EOF_OUTER'
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: benchmark-workload
  namespace: NAMESPACE_PLACEHOLDER
data:
  WORKLOAD_CONTENT_JSON: |
WORKLOAD_CONTENT_PLACEHOLDER
---
apiVersion: v1
kind: Pod
metadata:
  name: POD_NAME_PLACEHOLDER
  namespace: NAMESPACE_PLACEHOLDER
  labels:
    app: llm-d-benchmark
spec:
  restartPolicy: Never
  containers:
  - name: benchmark
    workingDir: /app/vllm/benchmarks/multi_turn
    image: IMAGE_PLACEHOLDER
    command: ["/bin/sh", "-c"]
    args:
    - |
      set -e
      wget https://www.gutenberg.org/ebooks/1184.txt.utf-8 && \
      mv 1184.txt.utf-8 pg1184.txt && \
      python3 benchmark_serving_multi_turn.py \
      ARGS_PLACEHOLDER && \
      echo "Copying result files to mounted volume..." && \
      cp -v *.xlsx *.json /results 2>/dev/null || echo "No .xlsx or .json files to copy"
    volumeMounts:
    - name: workload
      mountPath: /workload
      readOnly: true
    - name: results
      mountPath: /results
  volumes:
  - name: workload
    configMap:
      name: benchmark-workload
  - name: results
    hostPath:
      path: OUTPUT_DIR_PLACEHOLDER
      type: DirectoryOrCreate
EOF_OUTER

# Read workload content and indent it
WORKLOAD_CONTENT=$(cat "$WORKLOAD_FILE" | sed 's/^/    /')

# Build benchmark args YAML list
# ARGS_YAML=""
# for arg in "${BENCHMARK_ARGS[@]}"; do
#   ARGS_YAML+="    - \"$arg\"\n"
# done

for i in "${!WORKLOAD_FILE[@]}"; do
    if [[ "${WORKLOAD_FILE[i]}" == "--input-file" ]]; then
        if [[ $((i + 1)) -lt ${#WORKLOAD_FILE[@]} ]]; then
            WORKLOAD_CONTENT_JSON="${WORKLOAD_FILE[i+1]}"
            break
        else
            echo "Error: --input-file provided without a following value."
            exit 1
        fi
    fi
done

# keep the indent
ARGS_YAML="      ${BENCHMARK_ARGS[@]}"

# Replace placeholders
sed -i "s|NAMESPACE_PLACEHOLDER|$NAMESPACE|g" "$COMBINED_YAML"
sed -i "s|POD_NAME_PLACEHOLDER|$POD_NAME|g" "$COMBINED_YAML"
sed -i "s|IMAGE_PLACEHOLDER|$IMAGE|g" "$COMBINED_YAML"
sed -i "s|OUTPUT_DIR_PLACEHOLDER|$RESULTS_DIR|g" "$COMBINED_YAML"
sed -i "s|WORKLOAD_CONTENT_JSON|$(basename $WORKLOAD_FILE)|g" "$COMBINED_YAML"
sed -i "/WORKLOAD_CONTENT_PLACEHOLDER/r /dev/stdin" "$COMBINED_YAML" <<< "$WORKLOAD_CONTENT"
sed -i "/WORKLOAD_CONTENT_PLACEHOLDER/d" "$COMBINED_YAML"
sed -i "/ARGS_PLACEHOLDER/r /dev/stdin" "$COMBINED_YAML" <<< "$(echo -e "$ARGS_YAML")"
sed -i "/ARGS_PLACEHOLDER/d" "$COMBINED_YAML"

# Dry-run mode
if [ "$DRY_RUN" = true ]; then
  echo "=========================================="
  echo "DRY RUN: Multi-Turn Benchmark"
  echo "=========================================="
  echo ""
  echo "Would execute:"
  echo "  kubectl apply -f <combined-yaml> -n $NAMESPACE"
  echo ""
  echo "Generated YAML:"
  echo "----------------------------------------"
  cat "$COMBINED_YAML"
  echo "----------------------------------------"
  echo ""
  echo "Mock Output:"
  echo "configmap/benchmark-workload created"
  echo "pod/$POD_NAME created"
  echo ""
  echo "Benchmark execution (simulated):"
  echo "Processing conversations from $WORKLOAD_BASENAME..."
  echo "Completed: 95/100 successful"
  echo "Mean TTFT: 125ms"
  echo "Mean TPOT: 15ms"
  echo "Throughput: 25 conversations/sec"
  echo "=========================================="
  rm -f "$COMBINED_YAML"
  exit 0
fi

cat "$COMBINED_YAML"

# Execute benchmark
echo "Applying combined ConfigMap + Pod to namespace $NAMESPACE..."

if ! kubectl apply -f "$COMBINED_YAML" -n "$NAMESPACE"; then
  echo "ERROR: Failed to apply ConfigMap and Pod"
  rm -f "$COMBINED_YAML"
  exit 1
fi

echo "ConfigMap and Pod created successfully"
rm -f "$COMBINED_YAML"

# Wait for pod to start
echo "Waiting for pod to be ready..."
if ! kubectl wait --for=condition=Ready "pod/$POD_NAME" -n "$NAMESPACE" --timeout=300s 2>/dev/null; then
  echo "Warning: Pod did not become ready, but continuing to capture logs..."
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="$OUTPUT_DIR/benchmark_output.txt"

# Wait for pod completion; if timeout, pod may or may not be running
echo "Waiting for benchmark to complete (timeout: 600s)..."
kubectl wait --for=condition=Ready=false "pod/$POD_NAME" -n "$NAMESPACE" --timeout=600s 2>/dev/null

# Get final logs; these may be truncated if there are too many.
# Would need to switch to streaming instead.
kubectl logs "$POD_NAME" -n "$NAMESPACE" > "$OUTPUT_FILE" 2>&1 || true

# Save pod description for debugging
POD_DESC_FILE="$OUTPUT_DIR/pod_description.txt"
kubectl describe pod "$POD_NAME" -n "$NAMESPACE" > "$POD_DESC_FILE" 2>&1 || true

# Copy result files from pod's /results volume to host OUTPUT_DIR
echo "Copying result files from shared results folder to host..."
cp ${RESULTS_DIR}/* ${OUTPUT_DIR}

echo "Deleting ${RESULTS_DIR}"
rm -Rf ${RESULTS_DIR}/*

# Get exit code
EXIT_CODE=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[0].state.terminated.exitCode}' 2>/dev/null || echo "1")

if [ "$EXIT_CODE" != "0" ]; then
  echo "WARNING: Benchmark exited with code $EXIT_CODE"
  # Don't fail - let orchestrator handle it
fi

# Cleanup pod
echo "Cleaning up pod..."
kubectl delete pod "$POD_NAME" -n "$NAMESPACE" --wait=true 2>/dev/null || true

echo "Benchmark execution complete"
echo "Results saved to: $OUTPUT_FILE"

exit "$EXIT_CODE"
