#!/bin/bash
# Long document QA runner
# Executes long_doc_qa.py in a Kubernetes pod

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

# Generate pod name
POD_NAME="long-doc-qa-${RUN_LABEL}-$(date +%s)"

# Generate unique subdirectory for this benchmark run's results
RESULTS_SUBDIR="${NAMESPACE}-${POD_NAME}"

# Create manifests directory in output for user inspection
MANIFESTS_DIR="$OUTPUT_DIR/manifests"
mkdir -p "$MANIFESTS_DIR"

# Build benchmark args string for YAML
ARGS_STR=""
for arg in "${BENCHMARK_ARGS[@]}"; do
  ARGS_STR+="$arg "
done

# Generate Pod YAML
POD_YAML="$MANIFESTS_DIR/pod.yaml"

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
      # Create unique subdirectory for this run
      mkdir -p /results/$RESULTS_SUBDIR
      # Run benchmark (long_doc_qa.py outputs to current directory)
      python /utils/long_doc_qa.py ${ARGS_STR}
      # Copy results to mounted volume
      cp -v *.csv /results/$RESULTS_SUBDIR/ 2>/dev/null || echo "No .csv files to copy"
      cp -v *.png /results/$RESULTS_SUBDIR/ 2>/dev/null || echo "No .png files to copy"
    volumeMounts:
    - name: results
      mountPath: /results
    - name: utils
      mountPath: /utils
  volumes:
  - name: results
    # Placeholder - will be replaced by Kustomize patch with actual hostPath
    emptyDir: {}
  - name: utils
    hostPath:
      path: /mnt/rocm-icms-cache/utils
      type: DirectoryOrCreate
EOF

# Create kustomization with JSON patch to replace results volume
KUSTOMIZATION_YAML="$MANIFESTS_DIR/kustomization.yaml"
cat > "$KUSTOMIZATION_YAML" << 'EOF'
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
- pod.yaml

patches:
- target:
    kind: Pod
  patch: |-
    - op: replace
      path: /spec/volumes/0
      value:
        name: results
        hostPath:
          path: /mnt/rocm-icms-cache/benchmark-results
          type: DirectoryOrCreate
EOF

# Dry-run mode
if [ "$DRY_RUN" = true ]; then
  echo "=========================================="
  echo "DRY RUN: Long Document QA"
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
  echo "pod/$POD_NAME created"
  echo "Running long_doc_qa..."
  echo "Warmup round completed"
  echo "Query round completed"
  echo "Results saved to CSV files"
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

echo "Pod created successfully"

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
