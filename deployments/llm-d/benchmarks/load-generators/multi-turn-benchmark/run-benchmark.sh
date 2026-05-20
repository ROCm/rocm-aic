#!/bin/bash
# Multi-turn benchmark runner
# Executes vLLM benchmark in a Kubernetes pod with workload mounted via ConfigMap

set -euo pipefail

# Default values
IMAGE=""
NAMESPACE=""
WORKLOAD_FILE=""
OUTPUT_DIR=""
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
      echo "Usage: $0 --image IMAGE --namespace NS --workload-file FILE --output-dir DIR [--dry-run] -- <benchmark-args>"
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

# Validate workload file exists
if [ ! -f "$WORKLOAD_FILE" ]; then
  echo "ERROR: Workload file not found: $WORKLOAD_FILE"
  exit 1
fi

# Extract workload filename
WORKLOAD_BASENAME=$(basename "$WORKLOAD_FILE")

# Generate pod name (use timestamp for uniqueness)
POD_NAME="benchmark-runner-$(date +%s)"

# Create manifests directory in output for user inspection
MANIFESTS_DIR="$OUTPUT_DIR/manifests"
mkdir -p "$MANIFESTS_DIR"

# Generate ConfigMap YAML
CONFIGMAP_YAML="$MANIFESTS_DIR/configmap.yaml"
WORKLOAD_BASENAME=$(basename "$WORKLOAD_FILE")
WORKLOAD_CONTENT=$(cat "$WORKLOAD_FILE" | sed 's/^/    /')

cat > "$CONFIGMAP_YAML" << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: benchmark-workload
  namespace: $NAMESPACE
data:
  $WORKLOAD_BASENAME: |
$WORKLOAD_CONTENT
EOF

# Build benchmark args string
ARGS_STR="${BENCHMARK_ARGS[@]}"

# Generate unique subdirectory for this benchmark run's results
# This prevents conflicts when multiple benchmarks run in parallel
RESULTS_SUBDIR="${NAMESPACE}-${POD_NAME}"

# Generate Pod YAML
POD_YAML="$MANIFESTS_DIR/pod.yaml"
cat > "$POD_YAML" << EOF
apiVersion: v1
kind: Pod
metadata:
  name: $POD_NAME
  namespace: $NAMESPACE
  labels:
    app: llm-d-benchmark
spec:
  restartPolicy: Never
  containers:
  - name: benchmark
    workingDir: /app/vllm/benchmarks/multi_turn
    image: $IMAGE
    command: ["/bin/sh", "-c"]
    args:
    - |
      set -e
      # Create unique subdirectory for this run
      mkdir -p /results/$RESULTS_SUBDIR
      wget https://www.gutenberg.org/ebooks/1184.txt.utf-8 && \\
      mv 1184.txt.utf-8 pg1184.txt && \\
      python3 benchmark_serving_multi_turn.py \\
      $ARGS_STR && \\
      echo "Copying result files to mounted volume..." && \\
      cp -v *.xlsx *.json /results/$RESULTS_SUBDIR/ 2>/dev/null || echo "No .xlsx or .json files to copy"
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
    # Placeholder - will be replaced by Kustomize patch with actual hostPath
    emptyDir: {}
EOF

# Create kustomization with JSON patch to replace results volume
KUSTOMIZATION_YAML="$MANIFESTS_DIR/kustomization.yaml"
cat > "$KUSTOMIZATION_YAML" << 'EOF'
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
- configmap.yaml
- pod.yaml

patches:
- target:
    kind: Pod
  patch: |-
    - op: replace
      path: /spec/volumes/1
      value:
        name: results
        hostPath:
          path: /mnt/rocm-icms-cache/benchmark-results
          type: DirectoryOrCreate
EOF

# Dry-run mode
if [ "$DRY_RUN" = true ]; then
  echo "=========================================="
  echo "DRY RUN: Multi-Turn Benchmark"
  echo "=========================================="
  echo ""
  echo "Would execute:"
  echo "  kubectl apply -k $MANIFESTS_DIR -n $NAMESPACE"
  echo ""
  echo "Generated manifests saved to: $MANIFESTS_DIR"
  echo ""
  echo "Kustomized YAML (preview):"
  echo "----------------------------------------"
  kubectl kustomize "$MANIFESTS_DIR"
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
  exit 0
fi

# Show kustomized output for debugging
echo "Manifests saved to: $MANIFESTS_DIR"
echo "Kustomized YAML:"
kubectl kustomize "$MANIFESTS_DIR"

# Execute benchmark
echo "Applying kustomization to namespace $NAMESPACE..."

if ! kubectl apply -k "$MANIFESTS_DIR" -n "$NAMESPACE"; then
  echo "ERROR: Failed to apply kustomization"
  exit 1
fi

echo "ConfigMap and Pod created successfully"

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

# Extract the hostPath from the Kustomize patch
BASE_HOST_PATH=$(grep -A 5 "hostPath:" "$MANIFESTS_DIR/kustomization.yaml" | grep "path:" | head -1 | awk '{print $2}')

if [ -z "$BASE_HOST_PATH" ]; then
  echo "ERROR: Could not extract hostPath from kustomization.yaml"
  exit 1
fi

# Copy result files from shared hostPath to local OUTPUT_DIR
SHARED_RESULTS_PATH="${BASE_HOST_PATH}/$RESULTS_SUBDIR"
echo "Copying result files from $SHARED_RESULTS_PATH to $OUTPUT_DIR..."
if [ -d "$SHARED_RESULTS_PATH" ]; then
  cp -v "$SHARED_RESULTS_PATH"/* "$OUTPUT_DIR/" 2>/dev/null || echo "No result files to copy"
  # Clean up the shared directory
  rm -rf "$SHARED_RESULTS_PATH"
else
  echo "Warning: Shared results directory not found: $SHARED_RESULTS_PATH"
fi

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
