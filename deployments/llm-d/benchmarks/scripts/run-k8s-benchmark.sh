#!/bin/bash
# Generic Kubernetes Benchmark Runner
# Orchestrates benchmark execution in Kubernetes pods using manifest generators

set -euo pipefail

# Default values
TOOL_NAME=""
MANIFEST_GENERATOR=""
IMAGE=""
NAMESPACE=""
OUTPUT_DIR=""
RUN_LABEL="run1"
COMPLETION_TIMEOUT=1200
DRY_RUN=false
BENCHMARK_ARGS=()
EXTRA_MANIFEST_GENERATORS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --tool-name)
      TOOL_NAME="$2"
      shift 2
      ;;
    --manifest-generator)
      MANIFEST_GENERATOR="$2"
      shift 2
      ;;
    --extra-manifest-generator)
      EXTRA_MANIFEST_GENERATORS+=("$2")
      shift 2
      ;;
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
    --completion-timeout)
      COMPLETION_TIMEOUT="$2"
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
      echo "Usage: $0 --tool-name NAME --manifest-generator SCRIPT --image IMAGE --namespace NS --output-dir DIR [OPTIONS] -- <benchmark-args>"
      echo ""
      echo "Required arguments:"
      echo "  --tool-name NAME                Tool identifier (used in pod name)"
      echo "  --manifest-generator SCRIPT     Path to script that generates pod.yaml"
      echo "  --image IMAGE                   Container image to use"
      echo "  --namespace NS                  Kubernetes namespace"
      echo "  --output-dir DIR                Directory for outputs"
      echo ""
      echo "Optional arguments:"
      echo "  --run-label LABEL               Label for this run (default: run1)"
      echo "  --completion-timeout SECONDS    Timeout for benchmark completion (default: 1200)"
      echo "  --extra-manifest-generator PATH Additional manifest generator (e.g., for ConfigMaps)"
      echo "  --dry-run                       Show what would be executed without running"
      echo "  -- <benchmark-args>             Arguments to pass to the benchmark tool"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [ -z "$TOOL_NAME" ]; then
  echo "ERROR: --tool-name is required"
  exit 1
fi

if [ -z "$MANIFEST_GENERATOR" ]; then
  echo "ERROR: --manifest-generator is required"
  exit 1
fi

if [ ! -x "$MANIFEST_GENERATOR" ]; then
  echo "ERROR: Manifest generator not found or not executable: $MANIFEST_GENERATOR"
  exit 1
fi

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
POD_NAME="${TOOL_NAME}-${RUN_LABEL}-$(date +%s)"

# Generate unique subdirectory for this benchmark run's results
RESULTS_SUBDIR="${NAMESPACE}-${POD_NAME}"

# Create manifests directory and copy kustomize template
MANIFESTS_DIR="$OUTPUT_DIR/manifests-benchmark"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KUSTOMIZE_TEMPLATE="$SCRIPT_DIR/../load-generators/kustomize"

mkdir -p "$MANIFESTS_DIR"

# Copy kustomize template to manifests directory
if [ -d "$KUSTOMIZE_TEMPLATE" ]; then
  cp -r "$KUSTOMIZE_TEMPLATE"/* "$MANIFESTS_DIR/"
  echo "Copied kustomize template to manifests directory"
else
  echo "ERROR: Kustomize template directory not found: $KUSTOMIZE_TEMPLATE"
  exit 1
fi

# Build benchmark args string
ARGS_STR="${BENCHMARK_ARGS[*]}"

# Export variables for manifest generators
export POD_NAME
export IMAGE
export NAMESPACE
export RESULTS_SUBDIR
export ARGS_STR
export RUN_LABEL
export MANIFESTS_DIR  # Pass target directory to generators

# Generate extra manifests (e.g., ConfigMaps) in the manifests directory
for extra_gen in "${EXTRA_MANIFEST_GENERATORS[@]}"; do
  if [ ! -x "$extra_gen" ]; then
    echo "ERROR: Extra manifest generator not found or not executable: $extra_gen"
    exit 1
  fi

  EXTRA_MANIFEST_FILE="$MANIFESTS_DIR/$(basename "$extra_gen" .sh).yaml"
  echo "Generating extra manifest: $EXTRA_MANIFEST_FILE"

  if ! "$extra_gen" > "$EXTRA_MANIFEST_FILE"; then
    echo "ERROR: Failed to generate extra manifest from $extra_gen"
    exit 1
  fi

  # Add to kustomization resources
  RESOURCE_NAME="$(basename "$EXTRA_MANIFEST_FILE")"
  if ! grep -q "^- $RESOURCE_NAME$" "$MANIFESTS_DIR/kustomization.yaml"; then
    sed -i "/^resources:/a - $RESOURCE_NAME" "$MANIFESTS_DIR/kustomization.yaml"
  fi
done

# Generate Pod YAML in the manifests directory
POD_YAML="$MANIFESTS_DIR/pod.yaml"

echo "Generating pod manifest..."
if ! "$MANIFEST_GENERATOR" > "$POD_YAML"; then
  echo "ERROR: Failed to generate pod manifest"
  exit 1
fi

# Dry-run mode
if [ "$DRY_RUN" = true ]; then
  echo "=========================================="
  echo "DRY RUN: $TOOL_NAME Benchmark"
  echo "=========================================="
  echo ""
  echo "Pod Name: $POD_NAME"
  echo "Namespace: $NAMESPACE"
  echo "Image: $IMAGE"
  echo "Run Label: $RUN_LABEL"
  echo "Timeout: ${COMPLETION_TIMEOUT}s"
  echo "Benchmark Args: $ARGS_STR"
  echo ""
  echo "Would execute:"
  echo "  kubectl apply -k $MANIFESTS_DIR"
  echo ""
  echo "Generated manifests saved to: $MANIFESTS_DIR"
  echo "  - kustomization.yaml (from template)"
  echo "  - pod.yaml (generated)"
  if [ ${#EXTRA_MANIFEST_GENERATORS[@]} -gt 0 ]; then
    echo "  - Extra manifests (ConfigMaps, etc.)"
  fi
  echo ""
  echo "Kustomized YAML (preview):"
  echo "----------------------------------------"
  kubectl kustomize "$MANIFESTS_DIR" || true
  echo "----------------------------------------"
  echo ""
  echo "Benchmark execution (simulated):"
  echo "  Waiting for pod to be ready..."
  echo "  Running benchmark..."
  echo "  Benchmark completed"
  echo "  Collecting results..."
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

echo "Resources created successfully"

# Wait for pod to start
echo "Waiting for pod to be ready..."
if ! kubectl wait --for=condition=Ready "pod/$POD_NAME" -n "$NAMESPACE" --timeout=300s 2>/dev/null; then
  echo "Warning: Pod did not become ready, but continuing to capture logs..."
fi

# Create output directory structure
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/benchmark_output_${RUN_LABEL}.log"

# Wait for pod completion
echo "Waiting for benchmark to complete (timeout: ${COMPLETION_TIMEOUT}s)..."
kubectl wait --for=condition=Ready=false "pod/$POD_NAME" -n "$NAMESPACE" --timeout="${COMPLETION_TIMEOUT}s" 2>/dev/null || true

# Get logs
echo "Retrieving logs..."
kubectl logs "$POD_NAME" -n "$NAMESPACE" > "$LOG_FILE" 2>&1 || true

# Save pod description for debugging
POD_DESC_FILE="$OUTPUT_DIR/pod_description_${RUN_LABEL}.txt"
kubectl describe pod "$POD_NAME" -n "$NAMESPACE" > "$POD_DESC_FILE" 2>&1 || true

# Extract the hostPath from the kustomization.yaml (if it exists)
KUSTOMIZATION_FILE="$MANIFESTS_DIR/kustomization.yaml"
if [ -f "$KUSTOMIZATION_FILE" ]; then
  BASE_HOST_PATH=$(grep -A 5 "hostPath:" "$KUSTOMIZATION_FILE" 2>/dev/null | grep "path:" | head -1 | awk '{print $2}' || echo "")

  if [ -n "$BASE_HOST_PATH" ]; then
    # Copy result files from shared hostPath to local OUTPUT_DIR
    SHARED_RESULTS_PATH="${BASE_HOST_PATH}/$RESULTS_SUBDIR"
    echo "Copying result files from $SHARED_RESULTS_PATH to $OUTPUT_DIR..."
    if [ -d "$SHARED_RESULTS_PATH" ]; then
      cp -v "$SHARED_RESULTS_PATH"/* "$OUTPUT_DIR/" 2>/dev/null || echo "No result files to copy"
      # Clean up the shared directory
      #TODO: better way to do this?
      sudo rm -rf "$SHARED_RESULTS_PATH"
    else
      echo "Warning: Shared results directory not found: $SHARED_RESULTS_PATH"
    fi
  fi
fi

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
echo "Results saved to: $OUTPUT_DIR"

exit "$EXIT_CODE"
