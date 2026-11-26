# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Infrastructure targets: monitoring, export tarball, internal helpers.

.PHONY: endpoints env-docs env-docs-check \
        export check-hf-token check-gds-slab check-stack require-lmcache \
        check-kv-backend prep-dirs prep-kvbench-dirs

# docs/ENVIRONMENT.md's tables are generated from the files that own the
# defaults; descriptions live in docs/environment-vars.txt.
env-docs:
	@$(PYTHON) "$(CURDIR)/scripts/gen-env-reference.py" --write

env-docs-check:
	@$(PYTHON) "$(CURDIR)/scripts/gen-env-reference.py" --check

# Only Grafana's port is published, and not always on the port make picked
# (Makefile), so ask Docker where it landed.
endpoints:
	@bash "$(CURDIR)/scripts/aic-endpoints.sh" list

export:
	@git -C "$(CURDIR)" rev-parse HEAD >/dev/null 2>&1 || { \
		echo "ERROR: not a git checkout; cannot enumerate sources" >&2; exit 1; }
	@cd "$(CURDIR)" && git ls-files -z --cached --others --exclude-standard \
		| tar --null --no-recursion --ignore-failed-read --owner=0 --group=0 \
			--transform='s|^|$(EXPORT_PREFIX)/|' \
			-czf "$(EXPORT_TARBALL)" -T -
	@echo "Wrote $(EXPORT_TARBALL)"

check-hf-token:
	@if [ -z "$$HF_TOKEN" ] && [ -n "$(HF_TOKEN_FILE)" ] && [ -r "$(HF_TOKEN_FILE)" ]; then \
		export HF_TOKEN="$$(tr -d '\r\n' < "$(HF_TOKEN_FILE)")"; \
	fi; \
	if [ -z "$$HF_TOKEN" ]; then \
		echo "ERROR: set HF_TOKEN or HF_TOKEN_FILE" >&2; exit 1; \
	fi

check-gds-slab:
	@if [ -z "$(GDS_SLAB_DATA)" ]; then \
		echo "ERROR: GDS_SLAB_DATA must be set for GDS L1 mode" >&2; exit 1; \
	fi

# Refuse to bring up a stack whose AIC_STACK another checkout's stack is using.
check-stack:
	@bash "$(CURDIR)/scripts/aic-endpoints.sh" check-stack "$(CURDIR)/docker"

# For targets that drive LMCache itself (its HTTP API, the GDS slab L1).
require-lmcache:
	@if [ "$(AIC_KV_BACKEND)" != "lmcache" ]; then \
		echo "ERROR: this target needs AIC_KV_BACKEND=lmcache (got $(AIC_KV_BACKEND))" >&2; exit 1; \
	fi

# umbp_standalone_server exits at startup when it has a master but no RDMA
# engine address; say so before the stack is half up.
check-kv-backend:
	@if [ "$(AIC_KV_BACKEND)" = "mori" ] && [ -n "$(strip $(UMBP_MASTER_ADDRESS))" ] \
	    && [ -z "$(strip $(UMBP_IO_ENGINE_HOST))" ]; then \
		echo "ERROR: UMBP_MASTER_ADDRESS=$(UMBP_MASTER_ADDRESS) needs UMBP_IO_ENGINE_HOST (the address of this node's RDMA interface)" >&2; exit 1; \
	fi

prep-dirs:
	@mkdir -p "$(NVME_DATA)" "$(NFS_DATA)" \
		"$(LOG)/lmcache" "$(LOG)/vllm" "$(LOG)/kvbench" \
		"$(LOG)/vllm-emulator" "$(LOG)/lmcache-emulator" \
		"$(HF_HOME)/hub" "$(HF_HOME)/datasets" "$(HF_HOME)/vllm" \
		"$(HF_HOME)/vllm_config" "$(HF_HOME)/torch" "$(HF_HOME)/torch_inductor" \
		"$(BENCH_LOGDIR)/results" "$(BENCH_LOGDIR)/plots"

prep-kvbench-dirs:
	@mkdir -p "$(LOG)/kvbench" \
		"$(HF_HOME)" \
		"$(BENCH_LOGDIR)/results" "$(BENCH_LOGDIR)/plots"
