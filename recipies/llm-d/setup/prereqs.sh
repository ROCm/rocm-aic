#!/bin/bash

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Auto-accept flag
AUTO_ACCEPT=false
if [[ "${1:-}" == "-y" ]]; then
    AUTO_ACCEPT=true
fi

# Track installation status
declare -a INSTALLED=()
declare -a TO_INSTALL=()

# Function to prompt for installation
prompt_install() {
    local tool=$1
    if $AUTO_ACCEPT; then
        return 0
    fi
    read -p "Install $tool? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 0
    fi
    return 1
}

# Function to remove item from TO_INSTALL array
remove_from_todo() {
    local item=$1
    local new_array=()
    for i in "${TO_INSTALL[@]}"; do
        if [[ "$i" != "$item" ]]; then
            new_array+=("$i")
        fi
    done
    TO_INSTALL=("${new_array[@]}")
}

# Function to handle errors
handle_error() {
    local failed_tool=$1
    echo -e "${RED}Error: Failed to install $failed_tool${NC}" >&2
    echo
    if [ ${#INSTALLED[@]} -gt 0 ]; then
        echo -e "${GREEN}Successfully installed:${NC}"
        printf '  - %s\n' "${INSTALLED[@]}"
    fi
    if [ ${#TO_INSTALL[@]} -gt 0 ]; then
        echo -e "${YELLOW}Remaining to install:${NC}"
        printf '  - %s\n' "${TO_INSTALL[@]}"
    fi
    exit 1
}

# Check which tools are missing
echo "Checking prerequisites..."
echo

NEED_JUST=false
NEED_HELM=false
NEED_HELMFILE=false

if ! command -v just &> /dev/null; then
    echo -e "${YELLOW}just not found${NC}"
    NEED_JUST=true
    TO_INSTALL+=("just")
else
    echo -e "${GREEN}just found${NC}"
fi

if ! command -v helm &> /dev/null; then
    echo -e "${YELLOW}helm not found${NC}"
    NEED_HELM=true
    TO_INSTALL+=("helm")
else
    echo -e "${GREEN}helm found${NC}"
fi

if ! command -v helmfile &> /dev/null; then
    echo -e "${YELLOW}helmfile not found${NC}"
    NEED_HELMFILE=true
    TO_INSTALL+=("helmfile")
else
    echo -e "${GREEN}helmfile found${NC}"
fi

echo

# Install just if needed
if $NEED_JUST; then
    if prompt_install "just"; then
        echo "Installing just..."
        if ! sudo apt update; then
            handle_error "just (apt update)"
        fi
        if ! sudo apt install -y just; then
            handle_error "just (apt install)"
        fi
        INSTALLED+=("just")
        remove_from_todo "just"
        echo -e "${GREEN}just installed successfully${NC}"
        echo
    else
        echo "Skipping just installation"
        echo
    fi
fi

# Install helm if needed
if $NEED_HELM; then
    if prompt_install "helm"; then
        echo "Installing helm..."
        if ! wget https://get.helm.sh/helm-v3.20.1-linux-amd64.tar.gz; then
            handle_error "helm (download)"
        fi
        if ! tar xzf helm-v3.20.1-linux-amd64.tar.gz; then
            rm -f helm-v3.20.1-linux-amd64.tar.gz
            handle_error "helm (extract)"
        fi
        if ! sudo mv linux-amd64/helm /usr/local/bin/helm; then
            rm -rf linux-amd64 helm-v3.20.1-linux-amd64.tar.gz
            handle_error "helm (install)"
        fi
        rm -rf linux-amd64 helm-v3.20.1-linux-amd64.tar.gz
        INSTALLED+=("helm")
        remove_from_todo "helm"
        echo -e "${GREEN}helm installed successfully${NC}"
        echo
    else
        echo "Skipping helm installation"
        echo
    fi
fi

# Install helmfile if needed
if $NEED_HELMFILE; then
    if prompt_install "helmfile"; then
        echo "Installing helmfile..."
        if ! wget https://github.com/helmfile/helmfile/releases/download/v1.3.2/helmfile_1.3.2_linux_amd64.tar.gz; then
            handle_error "helmfile (download)"
        fi
        if ! tar xzf helmfile_1.3.2_linux_amd64.tar.gz; then
            rm -f helmfile_1.3.2_linux_amd64.tar.gz
            handle_error "helmfile (extract)"
        fi
        if ! sudo mv helmfile /usr/local/bin/helmfile; then
            rm -f helmfile_1.3.2_linux_amd64.tar.gz
            handle_error "helmfile (install)"
        fi
        rm -f helmfile_1.3.2_linux_amd64.tar.gz
        INSTALLED+=("helmfile")
        remove_from_todo "helmfile"
        echo -e "${GREEN}helmfile installed successfully${NC}"
        echo
    else
        echo "Skipping helmfile installation"
        echo
    fi
fi

# Install helm plugins if helm is available
if command -v helm &> /dev/null; then
    echo "Checking helm plugins..."

    # Check and install helm-secrets
    if ! helm plugin list 2>/dev/null | grep -q "secrets"; then
        echo "Installing helm-secrets plugin..."
        if ! helm plugin install https://github.com/jkroepke/helm-secrets --version v4.6.5; then
            handle_error "helm-secrets plugin"
        fi
        INSTALLED+=("helm-secrets plugin")
        echo -e "${GREEN}helm-secrets plugin installed${NC}"
    else
        echo -e "${GREEN}helm-secrets plugin already installed${NC}"
    fi

    # Check and install helm-diff
    if ! helm plugin list 2>/dev/null | grep -q "diff"; then
        echo "Installing helm-diff plugin..."
        if ! helm plugin install https://github.com/databus23/helm-diff; then
            handle_error "helm-diff plugin"
        fi
        INSTALLED+=("helm-diff plugin")
        echo -e "${GREEN}helm-diff plugin installed${NC}"
    else
        echo -e "${GREEN}helm-diff plugin already installed${NC}"
    fi

    # Check and install helm-git
    if ! helm plugin list 2>/dev/null | grep -q "helm-git"; then
        echo "Installing helm-git plugin..."
        if ! helm plugin install https://github.com/aslafy-z/helm-git --version 0.14.3; then
            handle_error "helm-git plugin"
        fi
        INSTALLED+=("helm-git plugin")
        echo -e "${GREEN}helm-git plugin installed${NC}"
    else
        echo -e "${GREEN}helm-git plugin already installed${NC}"
    fi
    echo
fi

# Final summary
echo -e "${GREEN}All prerequisites are installed!${NC}"
if [ ${#INSTALLED[@]} -gt 0 ]; then
    echo
    echo -e "${GREEN}Successfully installed:${NC}"
    printf '  - %s\n' "${INSTALLED[@]}"
fi
echo

# Check if /usr/local/bin is in PATH
if [[ ":$PATH:" != *":/usr/local/bin:"* ]]; then
    echo -e "${YELLOW}WARNING: /usr/local/bin is not in your PATH${NC}"
    echo "Add the following to your ~/.bashrc or ~/.zshrc:"
    echo "  export PATH=\"/usr/local/bin:\$PATH\""
    echo
    echo "Then run:"
    echo "  source ~/.bashrc  # or source ~/.zshrc"
else
    echo -e "${GREEN}/usr/local/bin is already in your PATH${NC}"
fi
