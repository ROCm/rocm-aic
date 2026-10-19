# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Benchmark targets: cliff, kvbench, stress, plot, emulate, profile capture.

.PHONY: cliff plot stress-grafana kvbench-build kvbench-up kvbench-logs kvbench-down \
        cliff-kvbench-local test-emulate-local stress-emulate-local capture-profile-local \
        test-lmcache-p2p-local test-lmcache-p2p-spur

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


# ---- rocm-ernic P2P test (local, no GPU hardware required) -------------------
#
# Boots two QEMU KVM VMs using qemu-tool (pip install qemu-tool) and the
# vfio-user-ernic-2vm compose stack from qemu-minimal.  Each VM gets a
# rocm-ernic emulated ionic RDMA NIC; the ernic TCP mesh connects them so
# RDMA verbs traffic travels between guests over the same TCP connections.
#
# VM1 (primary): lmcache coordinator + lmcache server with P2P
# VM2 (secondary): lmcache server with P2P + vllm serve (llm-emu)
#
# The test sends the same prompt to VM2's vllm twice and asserts that
# lmcache_mp_p2p_load_count_total > 0 on VM2's lmcache metrics.
#
# Requires: /dev/kvm, docker, qemu-tool (auto-installed), HF_TOKEN.
#           Two qcow2 VM images in AIC_LMCACHE_P2P_VM_IMAGES_DIR (ionic-flavour).

AIC_LMCACHE_P2P_WORK_DIR          ?= /tmp/aic-lmcache-p2p-test
AIC_LMCACHE_P2P_VM_IMAGES_DIR     ?= /var/lib/qemu-tool/images
# Pinned ionic-flavour qcow2 image from batesste-ci-images.
# The -qcow2 tag is an ORAS artifact (zstd qcow2); the bare tag is the OCI
# FROM-scratch payload image for docker create/cp extraction.
AIC_LMCACHE_P2P_QCOW2_IMAGE       ?= docker.io/sbates130272/batesste-ci-images-ubuntu-qcow2-gen-ionic:20260929.g2bdbd16-vm.resolute-ionic-qm.54cc234
AIC_LMCACHE_P2P_VM1_NAME          ?= qemu-minimal
AIC_LMCACHE_P2P_VM2_NAME          ?= qemu-minimal-2
AIC_LMCACHE_P2P_VM1_SSH_PORT      ?= 12230
AIC_LMCACHE_P2P_VM2_SSH_PORT      ?= 12231
AIC_LMCACHE_P2P_VM_VCPUS          ?= 4
AIC_LMCACHE_P2P_VM_MEM_MB         ?= 4096
AIC_LMCACHE_P2P_READY_S           ?= 300
AIC_LMCACHE_P2P_MODEL             ?= HuggingFaceTB/SmolLM2-135M-Instruct
AIC_LMCACHE_P2P_LMCACHE_IMAGE_REF ?= $(LMCACHE_IMAGE_REF)
AIC_LMCACHE_P2P_VLLM_IMAGE_REF    ?= $(IMAGE_REF)
AIC_LMCACHE_P2P_LMCACHE_L1_SIZE_GB ?= 0.5
AIC_LMCACHE_P2P_LMCACHE_PORT      ?= 6555
AIC_LMCACHE_P2P_COORD_PORT        ?= 9300
AIC_LMCACHE_P2P_PORT              ?= 18200
AIC_LMCACHE_P2P_VLLM_PORT         ?= 8000
# VM IPs assigned to the ionic interface (192.168.200.<10*index>)
AIC_LMCACHE_P2P_VM1_IP            ?= 192.168.200.10
AIC_LMCACHE_P2P_VM2_IP            ?= 192.168.200.20
# Path inside each VM where the P2P compose file is placed
AIC_LMCACHE_P2P_COMPOSE_PATH      ?= /tmp/lmcache-p2p/docker-compose.yml
# SSH host for each VM.  Default: localhost (published ports).
# In CI (DinD) override to the compose service name after joining the network.
AIC_LMCACHE_P2P_VM1_SSH_HOST      ?= localhost
AIC_LMCACHE_P2P_VM2_SSH_HOST      ?= localhost
# Set to 1 to skip compose up (CI runs it as a separate step before joining network)
AIC_LMCACHE_P2P_SKIP_COMPOSE_UP   ?= 0

_AIC_LMCACHE_P2P_SSH_FLAGS = -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=5

test-lmcache-p2p-local: prep-dirs
	@test -c /dev/kvm || { echo "ERROR: /dev/kvm not found — KVM is required" >&2; exit 1; }
	@test -n "$(HF_TOKEN)" || { echo "ERROR: HF_TOKEN not set" >&2; exit 1; }
	@test -n "$(AIC_LMCACHE_P2P_LMCACHE_IMAGE_REF)" || { \
	    echo "ERROR: AIC_LMCACHE_P2P_LMCACHE_IMAGE_REF not set (or LMCACHE_IMAGE_REF)" >&2; exit 1; }
	@echo "=== test-lmcache-p2p-local ==="
	@echo "  model=$(AIC_LMCACHE_P2P_MODEL)  VM1=$(AIC_LMCACHE_P2P_VM1_NAME):$(AIC_LMCACHE_P2P_VM1_SSH_PORT)  VM2=$(AIC_LMCACHE_P2P_VM2_NAME):$(AIC_LMCACHE_P2P_VM2_SSH_PORT)"
	@echo "[1/8] Ensuring qemu-tool is installed ..."
	@if ! qemu-tool --help >/dev/null 2>&1; then \
	    echo "  Installing qemu-tool via pipx ..."; \
	    if command -v pipx >/dev/null 2>&1; then \
	        PIPX_BIN_DIR=/usr/local/bin pipx install --force qemu-tool >/dev/null; \
	    else \
	        pip install --quiet --break-system-packages qemu-tool 2>/dev/null \
	            || pip install --quiet qemu-tool; \
	    fi; \
	fi
	@echo "[2/8] Extracting VM disk images from $(AIC_LMCACHE_P2P_QCOW2_IMAGE) ..."
	@mkdir -p "$(AIC_LMCACHE_P2P_VM_IMAGES_DIR)"
	@docker pull -q "$(AIC_LMCACHE_P2P_QCOW2_IMAGE)"
	@if [ ! -f "$(AIC_LMCACHE_P2P_VM_IMAGES_DIR)/$(AIC_LMCACHE_P2P_VM1_NAME).qcow2" ]; then \
	    echo "  Extracting $(AIC_LMCACHE_P2P_VM1_NAME).qcow2 ..."; \
	    _cid=$$(docker create "$(AIC_LMCACHE_P2P_QCOW2_IMAGE)"); \
	    mkdir -p "$(AIC_LMCACHE_P2P_WORK_DIR)/qcow2-tmp"; \
	    docker cp "$$_cid:/output/." "$(AIC_LMCACHE_P2P_WORK_DIR)/qcow2-tmp" 2>/dev/null \
	        || docker cp "$$_cid:/." "$(AIC_LMCACHE_P2P_WORK_DIR)/qcow2-tmp"; \
	    docker rm "$$_cid" >/dev/null; \
	    _qcow2=$$(find "$(AIC_LMCACHE_P2P_WORK_DIR)/qcow2-tmp" -name '*.qcow2' | head -1); \
	    cp "$$_qcow2" "$(AIC_LMCACHE_P2P_VM_IMAGES_DIR)/$(AIC_LMCACHE_P2P_VM1_NAME).qcow2"; \
	    cp "$$_qcow2" "$(AIC_LMCACHE_P2P_VM_IMAGES_DIR)/$(AIC_LMCACHE_P2P_VM2_NAME).qcow2"; \
	    rm -rf "$(AIC_LMCACHE_P2P_WORK_DIR)/qcow2-tmp"; \
	    echo "  Disk images extracted"; \
	else \
	    echo "  $(AIC_LMCACHE_P2P_VM1_NAME).qcow2 already present — skipping (delete $(AIC_LMCACHE_P2P_VM_IMAGES_DIR)/$(AIC_LMCACHE_P2P_VM1_NAME).qcow2 to refresh)"; \
	    if [ ! -f "$(AIC_LMCACHE_P2P_VM_IMAGES_DIR)/$(AIC_LMCACHE_P2P_VM2_NAME).qcow2" ]; then \
	        cp "$(AIC_LMCACHE_P2P_VM_IMAGES_DIR)/$(AIC_LMCACHE_P2P_VM1_NAME).qcow2" \
	           "$(AIC_LMCACHE_P2P_VM_IMAGES_DIR)/$(AIC_LMCACHE_P2P_VM2_NAME).qcow2"; \
	    fi; \
	fi
	@echo "[3/8] Starting 2-VM ernic compose stack ..."
	@if [ "$(AIC_LMCACHE_P2P_SKIP_COMPOSE_UP)" = "1" ]; then \
	    echo "  Skipping compose up (AIC_LMCACHE_P2P_SKIP_COMPOSE_UP=1)"; \
	else \
	    VM_IMAGES_DIR="$(AIC_LMCACHE_P2P_VM_IMAGES_DIR)" \
	    VM1_NAME="$(AIC_LMCACHE_P2P_VM1_NAME)" \
	    VM2_NAME="$(AIC_LMCACHE_P2P_VM2_NAME)" \
	    VM1_SSH_PORT="$(AIC_LMCACHE_P2P_VM1_SSH_PORT)" \
	    VM2_SSH_PORT="$(AIC_LMCACHE_P2P_VM2_SSH_PORT)" \
	    VM_VCPUS="$(AIC_LMCACHE_P2P_VM_VCPUS)" \
	    VM_VMEM="$(AIC_LMCACHE_P2P_VM_MEM_MB)" \
	        qemu-tool compose \
	            --stack vfio-user-ernic-2vm \
	            --vm-name "$(AIC_LMCACHE_P2P_VM1_NAME)" \
	            --vm2-name "$(AIC_LMCACHE_P2P_VM2_NAME)" \
	            up -d; \
	fi
	@echo "[4/8] Waiting for SSH on both VMs (up to $(AIC_LMCACHE_P2P_READY_S)s) ..."
	@for _vmspec in \
	        "$(AIC_LMCACHE_P2P_VM1_SSH_PORT) $(AIC_LMCACHE_P2P_VM1_SSH_HOST)" \
	        "$(AIC_LMCACHE_P2P_VM2_SSH_PORT) $(AIC_LMCACHE_P2P_VM2_SSH_HOST)"; do \
	    _port=$$(echo "$$_vmspec" | awk '{print $$1}'); \
	    _host=$$(echo "$$_vmspec" | awk '{print $$2}'); \
	    _ready=0; \
	    for _i in $$(seq 1 $$(($(AIC_LMCACHE_P2P_READY_S)/5))); do \
	        ssh $(_AIC_LMCACHE_P2P_SSH_FLAGS) -p "$$_port" "ubuntu@$$_host" true 2>/dev/null \
	            && { _ready=1; break; }; \
	        sleep 5; \
	    done; \
	    [ "$$_ready" = "1" ] || { \
	        echo "FAIL: VM on port $$_port not SSH-reachable after $(AIC_LMCACHE_P2P_READY_S)s" >&2; \
	        qemu-tool compose \
	            --stack vfio-user-ernic-2vm \
	            --vm-name "$(AIC_LMCACHE_P2P_VM1_NAME)" \
	            --vm2-name "$(AIC_LMCACHE_P2P_VM2_NAME)" \
	            logs --tail 40; \
	        qemu-tool compose \
	            --stack vfio-user-ernic-2vm \
	            --vm-name "$(AIC_LMCACHE_P2P_VM1_NAME)" \
	            --vm2-name "$(AIC_LMCACHE_P2P_VM2_NAME)" \
	            down; \
	        exit 1; \
	    }; \
	    echo "  SSH ready on :$$_port"; \
	done
	@echo "[5/8] Pushing LMCache P2P compose file and guest setup script into both VMs ..."
	@_compose_src="$(REPO_ROOT)/docker/compose/lmcache-p2p/docker-compose.yml"; \
	_setup_src="$(REPO_ROOT)/scripts/lmcache-p2p-guest-setup.sh"; \
	_setup_dst="$$(dirname $(AIC_LMCACHE_P2P_COMPOSE_PATH))/lmcache-p2p-guest-setup.sh"; \
	for _vmspec in \
	        "$(AIC_LMCACHE_P2P_VM1_SSH_PORT) $(AIC_LMCACHE_P2P_VM1_SSH_HOST)" \
	        "$(AIC_LMCACHE_P2P_VM2_SSH_PORT) $(AIC_LMCACHE_P2P_VM2_SSH_HOST)"; do \
	    _port=$$(echo "$$_vmspec" | awk '{print $$1}'); \
	    _host=$$(echo "$$_vmspec" | awk '{print $$2}'); \
	    ssh $(_AIC_LMCACHE_P2P_SSH_FLAGS) -p "$$_port" "ubuntu@$$_host" \
	        "mkdir -p $$(dirname $(AIC_LMCACHE_P2P_COMPOSE_PATH))"; \
	    scp -o StrictHostKeyChecking=no -P "$$_port" \
	        "$$_compose_src" "ubuntu@$$_host:$(AIC_LMCACHE_P2P_COMPOSE_PATH)"; \
	    scp -o StrictHostKeyChecking=no -P "$$_port" \
	        "$$_setup_src" "ubuntu@$$_host:$$_setup_dst"; \
	done
	@echo "[6/8] Configuring VMs and starting LMCache P2P compose ..."
	@_rc=0; \
	for _idx in 1 2; do \
	    _port="$(AIC_LMCACHE_P2P_VM1_SSH_PORT)"; [ "$$_idx" = "2" ] && _port="$(AIC_LMCACHE_P2P_VM2_SSH_PORT)"; \
	    _host="$(AIC_LMCACHE_P2P_VM1_SSH_HOST)"; [ "$$_idx" = "2" ] && _host="$(AIC_LMCACHE_P2P_VM2_SSH_HOST)"; \
	    _role="primary";             [ "$$_idx" = "2" ] && _role="secondary"; \
	    _this_ip="$(AIC_LMCACHE_P2P_VM1_IP)";     [ "$$_idx" = "2" ] && _this_ip="$(AIC_LMCACHE_P2P_VM2_IP)"; \
	    echo "  VM$$_idx ($$_role) :$$_port $$_this_ip ..."; \
	    ssh $(_AIC_LMCACHE_P2P_SSH_FLAGS) -p "$$_port" "ubuntu@$$_host" \
	        "LMCACHE_P2P_ROLE=$$_role \
	         THIS_IP=$$_this_ip \
	         COORD_IP=$(AIC_LMCACHE_P2P_VM1_IP) \
	         COMPOSE_FILE=$(AIC_LMCACHE_P2P_COMPOSE_PATH) \
	         LMCACHE_IMAGE_REF=$(AIC_LMCACHE_P2P_LMCACHE_IMAGE_REF) \
	         VLLM_IMAGE_REF=$(AIC_LMCACHE_P2P_VLLM_IMAGE_REF) \
	         LMCACHE_L1_SIZE_GB=$(AIC_LMCACHE_P2P_LMCACHE_L1_SIZE_GB) \
	         LMCACHE_PORT=$(AIC_LMCACHE_P2P_LMCACHE_PORT) \
	         COORD_PORT=$(AIC_LMCACHE_P2P_COORD_PORT) \
	         P2P_PORT=$(AIC_LMCACHE_P2P_PORT) \
	         VLLM_PORT=$(AIC_LMCACHE_P2P_VLLM_PORT) \
	         VLLM_MODEL=$(AIC_LMCACHE_P2P_MODEL) \
	         HF_TOKEN='$(HF_TOKEN)' \
	         bash $$(dirname $(AIC_LMCACHE_P2P_COMPOSE_PATH))/lmcache-p2p-guest-setup.sh" \
	        2>&1 | sed "s/^/  [vm$$_idx] /" || _rc=1; \
	done; \
	[ "$$_rc" -eq 0 ] || { \
	    echo "FAIL: guest setup failed" >&2; \
	    $(MAKE) --no-print-directory _lmcache-p2p-cleanup; \
	    exit 1; \
	}
	@echo "[7/8] Waiting for vllm on VM2 (up to $(AIC_LMCACHE_P2P_READY_S)s) ..."
	@_ready=0; \
	for _i in $$(seq 1 $$(($(AIC_LMCACHE_P2P_READY_S)/5))); do \
	    if ssh $(_AIC_LMCACHE_P2P_SSH_FLAGS) -p "$(AIC_LMCACHE_P2P_VM2_SSH_PORT)" ubuntu@"$(AIC_LMCACHE_P2P_VM2_SSH_HOST)" \
	            "curl -fsS http://127.0.0.1:$(AIC_LMCACHE_P2P_VLLM_PORT)/health >/dev/null 2>&1"; then \
	        _ready=1; break; \
	    fi; \
	    sleep 5; \
	done; \
	if [ "$$_ready" != "1" ]; then \
	    echo "FAIL: vllm not ready after $(AIC_LMCACHE_P2P_READY_S)s" >&2; \
	    ssh $(_AIC_LMCACHE_P2P_SSH_FLAGS) -p "$(AIC_LMCACHE_P2P_VM2_SSH_PORT)" ubuntu@"$(AIC_LMCACHE_P2P_VM2_SSH_HOST)" \
	        "docker compose -f $(AIC_LMCACHE_P2P_COMPOSE_PATH) --profile secondary logs --tail 30" 2>/dev/null \
	        | sed 's/^/  [vllm] /'; \
	    $(MAKE) --no-print-directory _lmcache-p2p-cleanup; \
	    exit 1; \
	fi; \
	echo "  vllm ready"
	@echo "[8/8] Running P2P test ..."
	@_rc=0; \
	_prompt=$$(python3 -c "print('AMD ROCm ' * 64)"); \
	_body="{\"model\":\"$(AIC_LMCACHE_P2P_MODEL)\",\"prompt\":\"$$_prompt\",\"max_tokens\":4,\"temperature\":0}"; \
	echo "  Sending warm-up request (populates VM1 cache via VM2 vllm) ..."; \
	ssh $(_AIC_LMCACHE_P2P_SSH_FLAGS) -p "$(AIC_LMCACHE_P2P_VM2_SSH_PORT)" ubuntu@"$(AIC_LMCACHE_P2P_VM2_SSH_HOST)" \
	    "curl -sS http://127.0.0.1:$(AIC_LMCACHE_P2P_VLLM_PORT)/v1/completions \
	         -H 'Content-Type: application/json' \
	         -d '$$_body'" 2>&1 | sed 's/^/  [pass1] /'; \
	echo "  Sending second request (should trigger P2P hit) ..."; \
	ssh $(_AIC_LMCACHE_P2P_SSH_FLAGS) -p "$(AIC_LMCACHE_P2P_VM2_SSH_PORT)" ubuntu@"$(AIC_LMCACHE_P2P_VM2_SSH_HOST)" \
	    "curl -sS http://127.0.0.1:$(AIC_LMCACHE_P2P_VLLM_PORT)/v1/completions \
	         -H 'Content-Type: application/json' \
	         -d '$$_body'" 2>&1 | sed 's/^/  [pass2] /'; \
	echo "  Collecting metrics from VM2 lmcache ..."; \
	_metrics=$$(ssh $(_AIC_LMCACHE_P2P_SSH_FLAGS) -p "$(AIC_LMCACHE_P2P_VM2_SSH_PORT)" ubuntu@"$(AIC_LMCACHE_P2P_VM2_SSH_HOST)" \
	    "curl -sS http://127.0.0.1:$(AIC_LMCACHE_P2P_LMCACHE_PORT)/metrics 2>/dev/null \
	     || curl -sS http://127.0.0.1:19090/metrics 2>/dev/null \
	     || echo no_metrics"); \
	printf '%s\n' "$$_metrics" \
	    | grep -E 'lmcache_mp_(p2p|remote|local|l1|l2)' \
	    | grep -v '^#' | sed 's/^/  [metrics] /'; \
	_p2p_hits=$$(printf '%s\n' "$$_metrics" \
	    | grep -oE 'lmcache_mp_(p2p_load|remote_hit)_count_total[[:space:]]+[0-9.]+' \
	    | awk '{s+=$$NF} END {printf "%d", s+0}'); \
	_p2p_hits=$${_p2p_hits:-0}; \
	if [ "$$_p2p_hits" -gt 0 ]; then \
	    echo "PASS: P2P hit count = $$_p2p_hits"; \
	else \
	    echo "WARN: P2P hit count = 0 (RDMA path may need ionic NIC config; check lmcache logs)" >&2; \
	    _rc=1; \
	fi; \
	$(MAKE) --no-print-directory _lmcache-p2p-cleanup; \
	[ "$$_rc" -eq 0 ] \
	    && echo "=== test-lmcache-p2p-local PASSED ===" \
	    || { echo "=== test-lmcache-p2p-local FAILED ===" >&2; exit 1; }

.PHONY: _lmcache-p2p-cleanup
_lmcache-p2p-cleanup:
	@VM_IMAGES_DIR="$(AIC_LMCACHE_P2P_VM_IMAGES_DIR)" \
	VM1_NAME="$(AIC_LMCACHE_P2P_VM1_NAME)" \
	VM2_NAME="$(AIC_LMCACHE_P2P_VM2_NAME)" \
	    qemu-tool compose \
	        --stack vfio-user-ernic-2vm \
	        --vm-name "$(AIC_LMCACHE_P2P_VM1_NAME)" \
	        --vm2-name "$(AIC_LMCACHE_P2P_VM2_NAME)" \
	        down 2>/dev/null || true

# ---- SPUR target: LMCache P2P test via rocm-ernic VMs -------------------
#
# Submits a SPUR job that boots two QEMU KVM VMs with rocm-ernic ionic NICs
# (vfio-user-ernic-2vm compose stack) and runs tests/test_lmcache_p2p.py.
# KVM is available on SPUR nodes (confirmed); Docker runs natively so
# localhost:12230/12231 SSH works without DinD workarounds.
#
# Usage:
#   make test-lmcache-p2p-spur \
#     HF_TOKEN=$(cat ~/.cache/huggingface/token) \
#     AIC_LMCACHE_IMAGE=<lmcache-image>

AIC_LMCACHE_P2P_SPUR_GPUS    ?= 1
AIC_LMCACHE_P2P_SPUR_NODE    ?=
AIC_LMCACHE_P2P_SPUR_TIME    ?= 60
AIC_LMCACHE_P2P_SPUR_WORKDIR ?= /shared_nfs/$(shell id -un)

test-lmcache-p2p-spur:
	@echo "=== test-lmcache-p2p-spur ==="
	@echo "  Submitting to amd-spur (gpus=$(AIC_LMCACHE_P2P_SPUR_GPUS) time=$(AIC_LMCACHE_P2P_SPUR_TIME)m)"
	@sbatch \
	     --partition=amd-spur \
	     --gres=gpu:$(AIC_LMCACHE_P2P_SPUR_GPUS) \
	     --time=$(AIC_LMCACHE_P2P_SPUR_TIME) \
	     $(if $(AIC_LMCACHE_P2P_SPUR_NODE),--nodelist=$(AIC_LMCACHE_P2P_SPUR_NODE),) \
	     --chdir="$(AIC_LMCACHE_P2P_SPUR_WORKDIR)" \
	     --export=ALL,SLURM_SUBMIT_DIR="$(REPO_ROOT)",HF_TOKEN="$(HF_TOKEN)",AIC_LMCACHE_P2P_QCOW2_IMAGE="$(AIC_LMCACHE_P2P_QCOW2_IMAGE)",AIC_LMCACHE_P2P_READY_S="$(AIC_LMCACHE_P2P_READY_S)",AIC_LMCACHE_P2P_VM1_IP="$(AIC_LMCACHE_P2P_VM1_IP)",AIC_LMCACHE_P2P_VM2_IP="$(AIC_LMCACHE_P2P_VM2_IP)" \
	     "$(REPO_ROOT)/.slurm/run-lmcache-p2p-test.sbatch"
