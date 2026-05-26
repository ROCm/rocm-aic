#!/bin/bash
# vLLM Bench Serve Pod Manifest Generator
# Generates Kubernetes Pod YAML for vllm bench serve

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
    tool: vllm-bench-serve
spec:
  restartPolicy: Never
  containers:
  - name: benchmark
    image: ${IMAGE}
    command: ["/bin/sh", "-c"]
    args:
    - |
      # Create unique subdirectory for this run
      mkdir -p /results/${RESULTS_SUBDIR}
      echo "[DBG] mkdir /results/${RESULTS_SUBDIR}"
      # Run benchmark
      vllm bench serve ${ARGS_STR}
      # Copy results to mounted volume
      cp -v *.json /results/${RESULTS_SUBDIR}/ 2>/dev/null || echo "No .json files to copy"
    volumeMounts:
    - name: results
      mountPath: /results
  # This will be patched by kustomize with the actual hostPath
EOF
