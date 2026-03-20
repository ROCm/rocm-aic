# Common prerequisite verification for all deployments
# Import this file in deployment-specific justfiles with: import? "../../common/prerequisites.justfile"

# Verify kubectl is installed and can connect to cluster
verify-kubectl:
    @command -v kubectl >/dev/null 2>&1 || (echo "❌ kubectl not found - please install kubectl" && exit 1)
    @kubectl cluster-info >/dev/null 2>&1 || (echo "❌ Cannot connect to Kubernetes cluster" && exit 1)
    @echo "✅ kubectl OK"

# Verify helm is installed
verify-helm:
    @command -v helm >/dev/null 2>&1 || (echo "❌ helm not found - please install helm" && exit 1)
    @echo "✅ helm OK ($(helm version --short))"

# Verify helmfile is installed
verify-helmfile:
    @command -v helmfile >/dev/null 2>&1 || (echo "❌ helmfile not found - please install helmfile" && exit 1)
    @echo "✅ helmfile OK ($(helmfile --version))"

# Verify kustomize is installed
verify-kustomize:
    @command -v kustomize >/dev/null 2>&1 || (echo "❌ kustomize not found - please install kustomize" && exit 1)
    @echo "✅ kustomize OK ($(kustomize version --short 2>/dev/null || kustomize version))"

# Verify all common tools
verify-all: verify-kubectl verify-helm verify-helmfile
    @echo "✅ All common prerequisites verified"
