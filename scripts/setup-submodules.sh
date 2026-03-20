#!/usr/bin/env bash
# Initialize and update git submodules for rocm-icms

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "Initializing submodules..."

# Check if .gitmodules exists
if [ ! -f ".gitmodules" ]; then
    echo "❌ .gitmodules not found"
    exit 1
fi

# Initialize and update submodules
if [ -d "submodules/llm-d/.git" ]; then
    echo "✅ llm-d submodule already initialized"
    echo "   Updating to latest..."
    git submodule update --remote --merge submodules/llm-d
else
    echo "Initializing llm-d submodule..."
    git submodule update --init --recursive submodules/llm-d
fi

# Show submodule status
echo ""
echo "Submodule status:"
git submodule status

echo ""
echo "✅ Submodules initialized successfully"
echo ""
echo "llm-d location: ${REPO_ROOT}/submodules/llm-d"
