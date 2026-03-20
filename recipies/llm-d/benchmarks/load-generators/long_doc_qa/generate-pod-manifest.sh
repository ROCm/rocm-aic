#!/bin/bash
# Long Document QA Pod Manifest Generator
# Generates Kubernetes Pod YAML for long_doc_qa benchmark

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
    tool: long-doc-qa
spec:
  restartPolicy: Never
  containers:
  - name: benchmark
    image: ${IMAGE}
    command: ["/bin/sh", "-c"]
    args:
    - |
      # Download the long_doc_qa.py script
      curl -O https://raw.githubusercontent.com/LMCache/LMCache/65bf93f8f38e4b8d800bf71f1285b771beb5482c/benchmarks/long_doc_qa/long_doc_qa.py
      # Create unique subdirectory for this run
      mkdir -p /results/${RESULTS_SUBDIR}
      # Run benchmark (long_doc_qa.py outputs to current directory)
      python long_doc_qa.py ${ARGS_STR}
      # Copy results to mounted volume
      cp -v *.csv /results/${RESULTS_SUBDIR}/ 2>/dev/null || echo "No .csv files to copy"
      cp -v *.png /results/${RESULTS_SUBDIR}/ 2>/dev/null || echo "No .png files to copy"
    volumeMounts:
    - name: results
      mountPath: /results
  # This will be patched by kustomize with the actual hostPath
EOF
