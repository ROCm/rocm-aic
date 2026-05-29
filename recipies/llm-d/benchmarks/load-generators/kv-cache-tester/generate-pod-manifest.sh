#!/bin/bash
# KV Cache Tester Pod Manifest Generator
# Generates Kubernetes Pod YAML for kv-cache-tester variants
#
# Supported variants:
# - single_prompt_tester: Simple single-prompt tests
# - cache_rate_tester: Test various cache hit rates
# - working_set_tester: Test performance across different memory tiers
# - trace_replay_tester: Replay real agentic coding traces

# Expected environment variables:
# - POD_NAME: Name of the pod
# - IMAGE: Container image
# - NAMESPACE: Kubernetes namespace
# - RESULTS_SUBDIR: Subdirectory for results in shared volume
# - ARGS_STR: Benchmark arguments (first arg should be --variant <variant_name>)

# Parse variant from ARGS_STR (expected format: --variant <variant_name> ...)
VARIANT=$(echo "${ARGS_STR}" | sed -n 's/.*--variant[[:space:]]\+\([a-z_]*\).*/\1/p')

# Default to single_prompt_tester if no variant specified
if [ -z "${VARIANT}" ]; then
    VARIANT="single_prompt_tester"
fi

# Validate variant
case "${VARIANT}" in
    single_prompt_tester|cache_rate_tester|working_set_tester|trace_replay_tester)
        ;;
    *)
        echo "ERROR: Invalid variant '${VARIANT}'. Valid variants: single_prompt_tester, cache_rate_tester, working_set_tester, trace_replay_tester" >&2
        exit 1
        ;;
esac

# Remove --variant from ARGS_STR as it's used for script selection, not passed to Python
FILTERED_ARGS=$(echo "${ARGS_STR}" | sed 's/--variant[[:space:]]\+[a-z_]*//g')

cat << EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${POD_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: llm-d-benchmark
    tool: kv-cache-tester
    variant: ${VARIANT}
spec:
  restartPolicy: Never
  containers:
  - name: benchmark
    image: ${IMAGE}
    command: ["/bin/sh", "-c"]
    args:
    - |
      set -e
      git clone --recursive https://github.com/callanjfox/kv-cache-tester.git
      cd kv-cache-tester
      git checkout 1cc386a
      # Create unique subdirectory for this run
      mkdir -p /results/${RESULTS_SUBDIR}
      echo "mkdir /results/${RESULTS_SUBDIR}"
      echo "Running variant: ${VARIANT}"
      echo "Args: ${FILTERED_ARGS}"

      # Run the appropriate variant
      set +e
      python ${VARIANT}.py ${FILTERED_ARGS}
      status=$?
      set -e
      echo "Benchmark exited with code=\${status}"

      # Copy results to mounted volume
      echo "Copying result files to mounted volume..."
      cp -v /tmp/results/* /results/${RESULTS_SUBDIR}/ 2>/dev/null || echo "No results files to copy"

      exit \${status}
    volumeMounts:
    - name: results
      mountPath: /results
    - name: inputs # use 'trace_directory' to set path to
      mountPath: /inputs
      readOnly: true
  volumes:
  - name: inputs
    hostPath:
      path: /mnt/rocm-icms-cache/benchmarks-inputs
  # This will be patched by kustomize with the actual hostPath
EOF
