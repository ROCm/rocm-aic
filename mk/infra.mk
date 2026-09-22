# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Infrastructure targets: monitoring, export tarball, internal helpers.

.PHONY: monitoring-up monitoring-down monitoring-logs \
        export check-hf-token check-gds-slab prep-dirs prep-kvbench-dirs

monitoring-up: ensure-compose
	@mkdir -p "$(AIC_METRICS_DIR)"
	PROM_UID="$$(id -u)" PROM_GID="$$(id -g)" AIC_HSA_SNOOP_PID_MODE="$${AIC_HSA_SNOOP_PID_MODE:-host}" \
		$(MON_COMPOSE) $(_MON_PROFILE) up -d
	@echo "Prometheus up on :9090  (TSDB -> $(AIC_METRICS_DIR))"

monitoring-down:
	AIC_HSA_SNOOP_PID_MODE="$${AIC_HSA_SNOOP_PID_MODE:-host}" $(MON_COMPOSE) $(_MON_PROFILE) down

monitoring-logs:
	$(MON_COMPOSE) logs -f prometheus

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

prep-dirs:
	@mkdir -p "$(NVME_DATA)" "$(NFS_DATA)" \
		"$(LOG)/lmcache" "$(LOG)/vllm" "$(LOG)/kvbench" \
		"$(HF_HOME)/hub" "$(HF_HOME)/datasets" "$(HF_HOME)/vllm" \
		"$(HF_HOME)/vllm_config" "$(HF_HOME)/torch" "$(HF_HOME)/torch_inductor" \
		"$(BENCH_LOGDIR)/results" "$(BENCH_LOGDIR)/plots"

prep-kvbench-dirs:
	@mkdir -p "$(LOG)/kvbench" \
		"$(HF_HOME)" \
		"$(BENCH_LOGDIR)/results" "$(BENCH_LOGDIR)/plots"
