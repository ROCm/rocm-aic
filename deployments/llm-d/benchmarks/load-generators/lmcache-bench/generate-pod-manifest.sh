#!/bin/bash
# LMCache Bench Pod Manifest Generator
# Generates Kubernetes Pod YAML for lmcache bench

# Expected environment variables:
# - POD_NAME: Name of the pod
# - IMAGE: Container image
# - NAMESPACE: Kubernetes namespace
# - RESULTS_SUBDIR: Subdirectory for results (unused by lmcache-bench, only uses logs)
# - ARGS_STR: Benchmark arguments

cat << EOF
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
