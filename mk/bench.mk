# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Benchmark targets: cliff, kvbench, stress, plot, emulate, profile capture.

.PHONY: cliff plot stress-grafana kvbench-build kvbench-up kvbench-logs kvbench-down \
        cliff-kvbench-local test-emulate-local stress-emulate-local capture-profile-local

cliff: prep-dirs
	@test -n "$(BENCH_MODEL)" || { \
		echo "ERROR: set BENCH_MODEL or VLLM_MODEL to the served model name" >&2; exit 1; }
	$(PYTHON) "$(CURDIR)/benchmarks/run_cliff.py" \
		--endpoint "$(BENCH_ENDPOINT)" \
		--model "$(BENCH_MODEL)" \
		--arm "$(BENCH_ARM)" \
		--isl "$(BENCH_ISL)" \
		--shared-prefix-tokens "$(BENCH_SHARED_TOK)" \
		--concurrencies "$(BENCH_CONCUR)" \
		--iters "$(BENCH_ITERS)" \
		--warmup-iters 1 \
		--out "$(BENCH_OUT)"
	@echo "Results written to $(BENCH_OUT)"

stress-grafana: prep-dirs
	@test -n "$(BENCH_MODEL)" || { \
		echo "ERROR: set BENCH_MODEL or VLLM_MODEL to the served model name" >&2; exit 1; }
	@docker inspect aic-client > /dev/null 2>&1 || { \
		echo "ERROR: aic-client container not running — start with 'make up'" >&2; exit 1; }
	@echo "Stressing stack for Grafana — watch http://localhost:3000  (ctrl-c to stop)"
	@echo "  endpoint: http://aic-vllm-gpu0:8000 (via aic-client)"
	@while true; do \
		docker exec aic-client \
			python3 -u benchmarks/run_cliff.py \
			--endpoint http://aic-vllm-gpu0:8000 \
			--model "$(BENCH_MODEL)" \
			--arm kvd_v2 \
			--isl 1024 \
			--shared-prefix-tokens 896 \
			--concurrencies 1,2,4,8 \
			--iters 5 \
			--warmup-iters 1 \
			--post-warmup-sleep-s 2 \
			--out /logs/manual/results/stress-$$(date +%Y%m%d-%H%M%S).csv; \
		echo "--- pass complete, restarting in 3s ---"; \
		sleep 3; \
	done

plot: prep-dirs
	$(PYTHON) "$(CURDIR)/benchmarks/plot_cliff.py" \
		--input "$(BENCH_LOGDIR)/results/" \
		--output-dir "$(BENCH_LOGDIR)/plots/"
	@echo "Charts written to $(BENCH_LOGDIR)/plots/"

kvbench-build: ensure-compose
	$(COMPOSE) --profile kvbench build kvbench

kvbench-up: ensure-compose prep-kvbench-dirs
	KVBENCH_IMAGE_REF="$(KVBENCH_IMAGE_REF)" $(COMPOSE) --profile kvbench up --build -d kvbench client
	@echo "KVBench is starting on http://localhost:$(KVBENCH_HOST_PORT) and http://aic-kvbench:$(KVBENCH_PORT) inside compose"

kvbench-logs:
	$(COMPOSE) --profile kvbench logs -f kvbench

kvbench-down:
	$(COMPOSE) --profile kvbench down --remove-orphans

cliff-kvbench-local: ensure-compose prep-kvbench-dirs
	@echo "=== cliff-kvbench-local: $(KVBENCH_IMAGE_REF) model=$${BENCH_MODEL:-$(KVBENCH_MODEL)} ==="
	@KVBENCH_IMAGE_REF="$(KVBENCH_IMAGE_REF)" $(COMPOSE) --profile kvbench up --build -d kvbench client
	@echo "Waiting up to $(KVBENCH_READY_S)s for KVBench /v1/models ..."
	@_ready=0; \
	for _i in $$(seq 1 $(KVBENCH_READY_S)); do \
	    if curl -fsS "http://localhost:$(KVBENCH_HOST_PORT)/v1/models" >/dev/null 2>&1; then _ready=1; break; fi; \
	    sleep 1; \
	done; \
	if [ "$$_ready" != "1" ]; then \
	    echo "FAIL: KVBench endpoint not ready after $(KVBENCH_READY_S)s" >&2; \
	    $(COMPOSE) --profile kvbench logs --tail 80 kvbench; \
	    $(COMPOSE) --profile kvbench down --remove-orphans >/dev/null 2>&1; \
	    exit 1; \
	fi; \
	for _i in $$(seq 1 60); do \
	    if docker exec aic-client python3 -c 'import openai' >/dev/null 2>&1; then break; fi; \
	    if [ "$$_i" = "60" ]; then \
	        echo "FAIL: aic-client did not finish installing benchmark deps" >&2; \
	        $(COMPOSE) logs --tail 80 client; \
	        $(COMPOSE) --profile kvbench down --remove-orphans >/dev/null 2>&1; \
	        exit 1; \
	    fi; \
	    sleep 1; \
	done; \
	out="/logs/manual/results/cliff-kvbench-$$(date +%Y%m%d-%H%M%S).csv"; \
	echo "Running cliff benchmark through the internal compose network -> $$out"; \
	docker exec aic-client python3 -u benchmarks/run_cliff.py \
	    --endpoint "http://aic-kvbench:$(KVBENCH_PORT)" \
	    --model "$${BENCH_MODEL:-$(KVBENCH_MODEL)}" \
	    --arm kvbench \
	    --isl "$(BENCH_ISL)" \
	    --shared-prefix-tokens "$(BENCH_SHARED_TOK)" \
	    --concurrencies "$(BENCH_CONCUR)" \
	    --iters "$(BENCH_ITERS)" \
	    --warmup-iters 0 \
	    --request-timeout 10 \
	    --out "$$out"; \
	echo "Results written to $${out#/logs/}"

# ---- Emulate targets (local, no Slurm) ----------------------------------------

AIC_EMULATE_MODEL     ?= Qwen/Qwen3-8B
AIC_EMULATE_READY_S   ?= 120
AIC_EMULATE_STRESS_CONCUR ?= 1,4,8,16
AIC_EMULATE_STRESS_ISL    ?= 512
AIC_EMULATE_STRESS_OSL    ?= 128
AIC_EMULATE_STRESS_ITERS  ?= 5

test-emulate-local: ensure-compose prep-dirs
	@echo "=== test-emulate-local: IMAGE_REF=$(IMAGE_REF) model=$(AIC_EMULATE_MODEL) ==="
	@VLLM_MODEL="$(AIC_EMULATE_MODEL)" IMAGE_REF="$(IMAGE_REF)" $(COMPOSE) --profile emulate up -d vllm-emulator
	@echo "Waiting up to $(AIC_EMULATE_READY_S)s for /health (engine fully ready) ..."
	@_ready=0; \
	for _i in $$(seq 1 $$(($(AIC_EMULATE_READY_S)/5))); do \
	    if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then _ready=1; break; fi; \
	    if [ -z "$$(docker ps -q -f name=aic-vllm-emulator)" ]; then \
	        echo "FAIL: emulator container exited during startup" >&2; \
	        $(COMPOSE) --profile emulate logs --tail 60 vllm-emulator; \
	        $(COMPOSE) --profile emulate down --remove-orphans >/dev/null 2>&1; \
	        exit 1; \
	    fi; \
	    sleep 5; \
	done; \
	if [ "$$_ready" != "1" ]; then \
	    echo "FAIL: endpoint not ready after $(AIC_EMULATE_READY_S)s" >&2; \
	    $(COMPOSE) --profile emulate logs --tail 80 vllm-emulator; \
	    $(COMPOSE) --profile emulate down --remove-orphans >/dev/null 2>&1; \
	    exit 1; \
	fi; \
	sleep 5; \
	echo "Endpoint ready — sending completion ..."; \
	resp=$$(curl -sS http://localhost:8000/v1/completions \
	    -H 'Content-Type: application/json' \
	    -d '{"model":"$(AIC_EMULATE_MODEL)","prompt":"Hello","max_tokens":16,"temperature":0}' 2>&1); \
	echo "Response: $$resp"; \
	rc=0; \
	tokens=$$(printf '%s' "$$resp" | grep -oE '"completion_tokens"[[:space:]]*:[[:space:]]*[0-9]+' | grep -oE '[0-9]+$$'); \
	if [ -n "$$tokens" ] && [ "$$tokens" -gt 0 ]; then \
	    echo "OK: emulated engine generated $$tokens tokens"; \
	else \
	    echo "FAIL: no tokens generated" >&2; rc=1; \
	fi; \
	logfile=$(LOG)/emulate-check.log; \
	mkdir -p "$(LOG)"; \
	$(COMPOSE) --profile emulate logs --no-color --no-log-prefix vllm-emulator > "$$logfile" 2>&1 || true; \
	if grep -q '\[ExecutorEmulatorHook\] Enabled' "$$logfile"; then \
	    echo "OK: executor hook active"; \
	else \
	    echo "FAIL: executor hook never activated (is VLLM_EMULATOR_ENABLE_ORACLE=1?)" >&2; rc=1; \
	fi; \
	if grep -q '\[ExecutorHook\] step=' "$$logfile"; then \
	    echo "OK: steps served from the profile pack"; \
	else \
	    echo "FAIL: no emulated steps recorded in logs" >&2; rc=1; \
	fi; \
	$(COMPOSE) --profile emulate down --remove-orphans >/dev/null 2>&1; \
	[ "$$rc" -eq 0 ] && echo "=== test-emulate-local PASSED ===" || { echo "=== test-emulate-local FAILED ===" >&2; exit 1; }

stress-emulate-local: ensure-compose prep-dirs
	@echo "=== stress-emulate-local: $(IMAGE_REF) model=$(AIC_EMULATE_MODEL) ==="
	@mkdir -p "$(AIC_METRICS_DIR)"
	IMAGE_REF="$(IMAGE_REF)" VLLM_MODEL="$(AIC_EMULATE_MODEL)" \
	    PROM_UID="$$(id -u)" PROM_GID="$$(id -g)" \
	    $(COMPOSE) --profile emulate --profile monitoring up -d vllm-emulator prometheus
	@echo "Waiting for emulator endpoint ..."
	@for _i in $$(seq 1 24); do \
	    curl -fsS http://localhost:8000/health >/dev/null 2>&1 && break; \
	    sleep 5; \
	done
	@echo "Running sweep: ISL=$(AIC_EMULATE_STRESS_ISL) OSL=$(AIC_EMULATE_STRESS_OSL) c=$(AIC_EMULATE_STRESS_CONCUR) x$(AIC_EMULATE_STRESS_ITERS)"
	docker exec aic-vllm-emulator vllm bench serve \
	    --host localhost --port 8000 \
	    --model "$(AIC_EMULATE_MODEL)" \
	    --dataset-name random \
	    --random-input-len "$(AIC_EMULATE_STRESS_ISL)" \
	    --random-output-len "$(AIC_EMULATE_STRESS_OSL)" \
	    --num-prompts $$(($(AIC_EMULATE_STRESS_ITERS) * 64)) \
	    --max-concurrency "$$(echo $(AIC_EMULATE_STRESS_CONCUR) | tr ',' '\n' | sort -rn | head -1)" \
	    --percentile-metrics ttft,tpot,itl,e2el \
	    --ignore-eos 2>&1 | tail -30
	@echo ""
	@echo "=== Key vLLM /metrics ==="
	@curl -s http://localhost:8000/metrics | \
	    grep -E "^vllm:(num_requests|e2e_request_latency|request_prompt_tokens|request_generation_tokens|gpu_cache_usage|request_success|request_failure)" | \
	    grep -v "^#" | sort | head -30
	@echo ""
	@echo "Prometheus at http://localhost:9090 — stack left running. Use 'make down' to stop."

# ---- Profile capture (local, requires /dev/kfd) --------------------------------

AIC_CAPTURE_MODEL     ?= Qwen/Qwen2.5-3B-Instruct
AIC_CAPTURE_HF_HOME   ?= $(HF_HOME)
AIC_CAPTURE_DIR       ?= $(CURDIR)/profiles/captures
AIC_CAPTURE_GPU       ?= $(GPU)
AIC_CAPTURE_GPU_UTIL  ?= 0.85
AIC_CAPTURE_MAX_MODEL_LEN     ?= 4096
AIC_CAPTURE_MAX_BATCHED_TOKENS ?= 2048
AIC_CAPTURE_SWEEP ?= 128,16,1,12 128,16,8,64 512,64,1,8 512,64,4,32 \
                     1024,64,1,8 1024,64,4,32 1024,64,16,64 \
                     2048,64,1,6 2048,64,4,24 4096,64,1,4 4096,64,4,16
AIC_CAPTURE_WARMUP_SKIP ?= 5

capture-profile-local: prep-dirs
	@test -e /dev/kfd || { echo "ERROR: /dev/kfd not found — GPU not accessible here"; exit 1; }
	@test -n "$(ROCM_ARCH)" || { echo "ERROR: ROCM_ARCH empty" >&2; exit 1; }
	@mkdir -p "$(AIC_CAPTURE_DIR)/bench"
	@stamp=$$(date +%Y%m%d-%H%M%S); \
	model_tag=$$(echo '$(AIC_CAPTURE_MODEL)' | tr '/' '-'); \
	trace_file="step-trace-$${model_tag}-$${stamp}.jsonl"; \
	pack_name="$${model_tag}-$${stamp}.json"; \
	echo "=== capture-profile-local: IMAGE_REF=$(IMAGE_REF) model=$(AIC_CAPTURE_MODEL) arch=$(ROCM_ARCH) ==="; \
	echo "Trace -> $(AIC_CAPTURE_DIR)/$${trace_file}"; \
	docker run -d --name aic-vllm-capture \
	    --device /dev/kfd --device /dev/dri \
	    --network host --ipc host \
	    --cap-add CAP_SYS_ADMIN --cap-add SYS_PTRACE \
	    --security-opt seccomp=unconfined \
	    -v "$(AIC_CAPTURE_HF_HOME):/hf" \
	    -v "$(AIC_CAPTURE_DIR):/trace" \
	    -e HF_HOME=/hf -e HF_HUB_CACHE=/hf/hub \
	    -e HF_TOKEN="$${HF_TOKEN:-}" -e HF_HUB_OFFLINE=0 \
	    -e ROCR_VISIBLE_DEVICES="$(AIC_CAPTURE_GPU)" \
	    -e VLLM_ROCM_USE_AITER=1 \
	    -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:False \
	    -e PYTHONUNBUFFERED=1 \
	    -e VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
	    -e "VLLM_EMULATOR_STEP_TRACE_OUTPUT=/trace/$${trace_file}" \
	    "$(IMAGE_REF)" \
	    --model "$(AIC_CAPTURE_MODEL)" \
	    --host 0.0.0.0 --port 8000 \
	    --max-model-len "$(AIC_CAPTURE_MAX_MODEL_LEN)" \
	    --max-num-batched-tokens "$(AIC_CAPTURE_MAX_BATCHED_TOKENS)" \
	    --gpu-memory-utilization "$(AIC_CAPTURE_GPU_UTIL)" \
	    --no-enable-prefix-caching \
	    --attention-backend TRITON_ATTN \
	    --disable-access-log-for-endpoints "/health,/metrics,/v1/models" \
	    >/dev/null || { echo "FAIL: docker run failed" >&2; exit 1; }; \
	echo "Waiting for endpoint (weights download may take a few minutes) ..."; \
	_ready=0; for _i in $$(seq 1 60); do \
	    curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1 && { _ready=1; break; }; \
	    [ -z "$$(docker ps -q -f name=aic-vllm-capture)" ] && \
	        { docker logs --tail 40 aic-vllm-capture >&2; docker rm -f aic-vllm-capture >/dev/null 2>&1; exit 1; }; \
	    sleep 10; \
	done; \
	[ "$$_ready" != "1" ] && { echo "FAIL: endpoint never ready" >&2; docker rm -f aic-vllm-capture >/dev/null 2>&1; exit 1; }; \
	echo "Endpoint ready — running sweep ..."; \
	rc=0; _seed=0; \
	for point in $(AIC_CAPTURE_SWEEP); do \
	    _seed=$$((_seed+1)); \
	    isl=$$(echo "$$point" | cut -d, -f1); \
	    osl=$$(echo "$$point" | cut -d, -f2); \
	    conc=$$(echo "$$point" | cut -d, -f3); \
	    np=$$(echo "$$point" | cut -d, -f4); \
	    echo "--- isl=$$isl osl=$$osl c=$$conc n=$$np ---"; \
	    docker exec aic-vllm-capture vllm bench serve \
	        --host localhost --port 8000 \
	        --model "$(AIC_CAPTURE_MODEL)" \
	        --dataset-name random \
	        --random-input-len "$$isl" --random-output-len "$$osl" \
	        --num-prompts "$$np" --max-concurrency "$$conc" \
	        --ignore-eos --seed "$$_seed" \
	        --percentile-metrics ttft,tpot,itl,e2el \
	        --save-result --result-dir /trace/bench \
	        --result-filename "real-$${model_tag}-isl$${isl}-osl$${osl}-c$${conc}.json" \
	        2>&1 | sed 's/^/  [bench] /' || rc=1; \
	done; \
	echo "Sweep done (rc=$$rc); stopping server ..."; \
	docker stop -t 60 aic-vllm-capture >/dev/null 2>&1 || true; \
	sleep 3; \
	docker rm -f aic-vllm-capture >/dev/null 2>&1 || true; \
	[ -s "$(AIC_CAPTURE_DIR)/$$trace_file" ] || { echo "FAIL: trace not written" >&2; exit 1; }; \
	echo "Building profile pack ..."; \
	docker run --rm \
	    -v "$(AIC_CAPTURE_DIR):/trace" \
	    --entrypoint python3 "$(IMAGE_REF)" \
	    -m vllm_emulator.profile.build_serving_profile_filtered \
	    "/trace/$$trace_file" "/trace/$$pack_name" \
	    --warmup-skip "$(AIC_CAPTURE_WARMUP_SKIP)" \
	    2>&1 | sed 's/^/  [pack] /'; \
	echo "Pack: $(AIC_CAPTURE_DIR)/$$pack_name"; \
	echo "Copy it to profiles/ and add a .capture.txt sibling to use it in CI."
