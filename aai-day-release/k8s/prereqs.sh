#!/usr/bin/env bash
# Install prerequisites for the aai-day-release Kubernetes deployment.
# Tested on Ubuntu 22.04/24.04. Run as a user with sudo rights.
set -euo pipefail

# kubectl
if ! command -v kubectl &>/dev/null; then
  echo "Installing kubectl..."
  curl -fsSL "https://dl.k8s.io/release/$(curl -fsSL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    -o /tmp/kubectl
  sudo install -o root -g root -m 0755 /tmp/kubectl /usr/local/bin/kubectl
  echo "kubectl $(kubectl version --client --short 2>/dev/null || kubectl version --client) installed."
fi

# kustomize
if ! command -v kustomize &>/dev/null; then
  echo "Installing kustomize..."
  KUSTOMIZE_VERSION="v5.4.3"
  curl -fsSL "https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2F${KUSTOMIZE_VERSION}/kustomize_${KUSTOMIZE_VERSION}_linux_amd64.tar.gz" \
    | tar -xz -C /tmp
  sudo install -o root -g root -m 0755 /tmp/kustomize /usr/local/bin/kustomize
  echo "kustomize $(kustomize version) installed."
fi

# just
if ! command -v just &>/dev/null; then
  echo "Installing just..."
  curl -fsSL https://just.systems/install.sh | sudo bash -s -- --to /usr/local/bin
  echo "just $(just --version) installed."
fi

echo ""
echo "All prerequisites installed."
echo "Next steps:"
echo "  1. Configure your kubeconfig: export KUBECONFIG=~/.kube/config"
echo "  2. Verify cluster access: kubectl get nodes"
echo "  3. Register your HuggingFace token: just register-hf-token"
echo "  4. Deploy the quickstart flavor: just deploy-tp2"
