#!/usr/bin/env bash
# Verify Kubernetes cluster connectivity and readiness

set -euo pipefail

echo "Kubernetes Cluster Verification"
echo "==============================="
echo ""

# Check kubectl
echo -n "Checking kubectl... "
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found"
    exit 1
fi
echo "✅ $(kubectl version --client --short 2>/dev/null || kubectl version --client | head -1)"

# Check cluster connectivity
echo -n "Checking cluster connectivity... "
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Cannot connect to Kubernetes cluster"
    echo ""
    echo "Please ensure:"
    echo "  1. Kubernetes cluster is running"
    echo "  2. kubectl is configured correctly"
    echo "  3. Current context is set correctly"
    exit 1
fi
echo "✅ Connected"

# Show current context
echo ""
echo "Current context: $(kubectl config current-context)"

# Show cluster info
echo ""
echo "Cluster information:"
kubectl cluster-info

# Check node status
echo ""
echo "Node status:"
kubectl get nodes

# Check for AMD GPU nodes (optional)
echo ""
echo "Checking for AMD GPU nodes:"
if kubectl get nodes -o json | jq -r '.items[].status.capacity | select(.["amd.com/gpu"]) | .["amd.com/gpu"]' 2>/dev/null | grep -q .; then
    echo "✅ AMD GPU resources found:"
    kubectl get nodes -o custom-columns=NAME:.metadata.name,AMD_GPUS:.status.capacity.amd\.com/gpu
else
    echo "⚠️  No AMD GPU resources found on nodes"
    echo "   This is OK if you're testing without GPUs"
fi

echo ""
echo "✅ Cluster verification complete"
