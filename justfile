# ROCm-ICMS

# Default recipe - show available commands
default:
    @just --list

# Initialize submodule
setup-submodules:
    @echo "Initializing submodules..."
    @./scripts/setup-submodules.sh

# Update submodule to latest
update-submodules:
    @echo "Updating submodules..."
    @git submodule update --remote --merge
    @echo "✅ Submodules updated"

# Show git submodule status
submodule-status:
    @echo "Submodule status:"
    @git submodule status

# Quick health check of the repository setup
health-check:
    @echo "Repository Health Check"
    @echo "======================="
    @echo ""
    @echo "Submodules:"
    @if [ -f "submodules/llm-d/.git" ]; then \
        echo "  ✓ llm-d submodule initialized"; \
    else \
        echo "  ✗ llm-d submodule not initialized (run 'just setup-submodules')"; \
    fi
