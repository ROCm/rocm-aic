#!/bin/bash
# Multi-turn Benchmark Pod Manifest Generator
# Generates Kubernetes Pod YAML for multi-turn benchmark

# Expected environment variables:
# - POD_NAME: Name of the pod
# - IMAGE: Container image
# - NAMESPACE: Kubernetes namespace
# - RESULTS_SUBDIR: Subdirectory for results in shared volume
# - ARGS_STR: Benchmark arguments

cat << EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${POD_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: llm-d-benchmark
    tool: multi-turn-benchmark
spec:
  restartPolicy: Never
  containers:
  - name: benchmark
    workingDir: /app/vllm/benchmarks/multi_turn
    image: ${IMAGE}
    command: ["/bin/sh", "-c"]
    args:
    - |
      set -e
      pip install xlsxwriter
      # Create unique subdirectory for this run
      mkdir -p /results/${RESULTS_SUBDIR}
      wget https://www.gutenberg.org/ebooks/1184.txt.utf-8 && \\
      mv 1184.txt.utf-8 pg1184.txt && \\
      python3 benchmark_serving_multi_turn.py \\
      ${ARGS_STR} && \\
      echo "Copying result files to mounted volume..." && \\
      cp -v *.xlsx *.json /results/${RESULTS_SUBDIR}/ 2>/dev/null || echo "No .xlsx or .json files to copy"
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
  # This will be patched by kustomize with the actual hostPath
EOF
