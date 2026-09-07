#!/usr/bin/env bash
set -euo pipefail

# Runs from a workflow checkout on the self-hosted runner; SSHes to the SPUR head node (AIC_SPUR_HOST),
# submits a CPU-only srun job (no --gres=gpu), and:
#   1. Launches node-exporter, nvme-exporter, rdma-exporter, and Prometheus
#   2. Runs the vLLM emulator (backed by the existing rocm-aic image)
#   3. Issues 3 LLM prompts and asserts non-empty completions
#   4. Scrapes all active /metrics endpoints and generates a metrics HTML page
#
# The generated metrics-page/index.html is copied back to the runner working
# directory so the workflow can upload it as an artifact and deploy to gh-pages.
#
# Uses the rocm-aic image tarball left by spur-dist-build.sh for the same SHA
# (same TARBALL_DIR convention).  Set AIC_SMOKE_USE_REGISTRY=1 to pull from
# a registry instead.

SHA="${1:?usage: $0 <full-sha>}"
SHORT="${SHA:0:7}"
AIC_IMAGE_NAME="rocm-aic-ci-${SHORT}"
AIC_SPUR_HOST="${AIC_SPUR_HOST:?AIC_SPUR_HOST must be set (e.g. via GitHub repo variable)}"
AIC_SPUR_HOST="${AIC_SPUR_HOST//[$'\t\r\n ']}"
AIC_SHARED_NFS="${AIC_SHARED_NFS:?AIC_SHARED_NFS must be set (e.g. via GitHub repo variable)}"
AIC_SPUR_CONTROLLER="${AIC_SPUR_CONTROLLER:?AIC_SPUR_CONTROLLER must be set (e.g. via GitHub repo variable)}"
AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT:-}"
AIC_SMOKE_USE_REGISTRY="${AIC_SMOKE_USE_REGISTRY:-0}"
AIC_SLURM_ACCOUNT="${AIC_SLURM_ACCOUNT:-}"
REPO="https://github.com/ROCm/rocm-aic.git"

ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "${AIC_SPUR_HOST}" env \
    SHA="${SHA}" \
    REPO="${REPO}" \
    AIC_IMAGE_NAME="${AIC_IMAGE_NAME}" \
    AIC_SHARED_NFS="${AIC_SHARED_NFS}" \
    AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT}" \
    AIC_SPUR_CONTROLLER="${AIC_SPUR_CONTROLLER}" \
    SPUR_CONTROLLER_ADDR="${AIC_SPUR_CONTROLLER}" \
    AIC_SLURM_ACCOUNT="${AIC_SLURM_ACCOUNT}" \
    AIC_SMOKE_USE_REGISTRY="${AIC_SMOKE_USE_REGISTRY}" \
    bash << 'REMOTE'
set -euo pipefail
export PATH="/usr/local/bin:${PATH}"

SHORT="${SHA:0:7}"
WORKDIR="$HOME/Projects/rocm-aic.${SHORT}"
CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT:-$HOME/Projects/rocm-aic-ci}"
TARBALL_DIR="${CI_STORAGE_ROOT}/images/aic-ci-${SHORT}"
AIC_METRICS_PAGE_DIR="${AIC_SHARED_NFS}/rocm-aic/metrics-pages-${SHORT}"
HF_HOME="${AIC_SHARED_NFS}/huggingface"

cleanup() {
    echo "=== Cleaning up ==="
    rm -rf "${WORKDIR}"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Clone the repo at this SHA so we have monitoring/ and docker/ locally
# ---------------------------------------------------------------------------
ACTUAL_SHA="$(git -C "${WORKDIR}" rev-parse HEAD 2>/dev/null || true)"
if [[ ! -d "${WORKDIR}" || "${ACTUAL_SHA}" != "${SHA}" ]]; then
    echo "=== (Re-)cloning ${REPO} at ${SHA} ==="
    rm -rf "${WORKDIR}"
    git clone --filter=blob:none --no-single-branch "${REPO}" "${WORKDIR}"
fi
cd "${WORKDIR}"
git checkout "${SHA}"

# Now that the tree is checked out, derive the version tag from the Dockerfile
# pins so this stage names the image exactly as spur-dist-build.sh did.  Falling
# back to :latest matches run-build-distribute.sh when the helper cannot resolve.
_aic_tag="$(bash "${WORKDIR}/docker/scripts/aic-image-tag.sh" 2>/dev/null || true)"
AIC_IMAGE="${AIC_IMAGE_NAME}:${_aic_tag:-latest}"
echo "=== Image ref for this SHA: ${AIC_IMAGE} ==="

# ---------------------------------------------------------------------------
# Submit a CPU-only srun job
# ---------------------------------------------------------------------------
SRUN_SCRIPT="$(mktemp "${WORKDIR}/aic-cpu-smoke-XXXXXX.sh")"

cat > "${SRUN_SCRIPT}" << 'SRUN_BODY'
#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${1}"
AIC_IMAGE="${2}"
AIC_METRICS_PAGE_DIR="${3}"
SHA="${4}"
TARBALL_DIR="${5}"
AIC_SMOKE_USE_REGISTRY="${6:-0}"
HF_HOME="${7}"
SHORT="${SHA:0:7}"

MON_DIR="${WORKDIR}/monitoring"
EMULATOR_COMPOSE="${WORKDIR}/docker/docker-compose.emulator.yml"
METRICS_DIR="${AIC_METRICS_PAGE_DIR}/prom-tsdb"
mkdir -p "${METRICS_DIR}" "${AIC_METRICS_PAGE_DIR}"

# ---------------------------------------------------------------------------
# Load or pull the rocm-aic image (docker is on the compute node, not the head node)
# ---------------------------------------------------------------------------
if [[ "${AIC_SMOKE_USE_REGISTRY}" == "1" ]]; then
    docker pull "${AIC_IMAGE}"
else
    echo "=== Loading image from ${TARBALL_DIR} ==="
    img_base="$(printf '%s' "${AIC_IMAGE}" | tr '/:' '--')"
    tarball="$(find "${TARBALL_DIR}" -maxdepth 1 -name "${img_base}-*.tar.zst" -o -name "${img_base}-*.tar.gz" -o -name "${img_base}-*.tar" 2>/dev/null | head -1)"
    [[ -n "${tarball}" ]] || { echo "ERROR: no tarball for ${AIC_IMAGE} in ${TARBALL_DIR}" >&2; ls -la "${TARBALL_DIR}" >&2 || true; exit 1; }
    case "${tarball}" in
        *.tar.zst) zstd -dc "${tarball}" | docker load ;;
        *.tar.gz)  gzip -dc "${tarball}" | docker load ;;
        *.tar)     docker load -i "${tarball}" ;;
    esac
fi

# ---------------------------------------------------------------------------
# Source shared monitoring helpers and configure for CPU-only smoke test
# ---------------------------------------------------------------------------
mkdir -p "${HF_HOME}"
# shellcheck source=monitoring/monitoring-lib.sh
source "${MON_DIR}/monitoring-lib.sh"

export AIC_IMAGE AIC_MONITORING=1 AIC_EXPORTERS=1 AIC_CPU_SMOKE=1
export MON_DIR MON_COMPOSE="${MON_DIR}/docker-compose.monitoring.yml"
export AIC_METRICS_DIR="${METRICS_DIR}"

cleanup() {
    echo "=== Stopping all containers ==="
    IMAGE_NAME="${AIC_IMAGE}" \
        docker compose -f "${EMULATOR_COMPOSE}" down --remove-orphans 2>/dev/null || true
    stop_monitoring
    rm -rf "${METRICS_DIR}"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 1. Pre-build fabric exporter images (nvme + rdma) from source
# ---------------------------------------------------------------------------
echo "=== Building fabric exporter images ==="
docker compose -f "${MON_COMPOSE}" \
    --profile exporters-cpu \
    build nvme-exporter rdma-exporter
export AIC_NVME_EXPORTER_IMAGE=aic-nvme-exporter:local
export AIC_RDMA_EXPORTER_IMAGE=aic-rdma-exporter:local

# ---------------------------------------------------------------------------
# 2. Start monitoring stack (prometheus + node-exporter + nvme + rdma)
# ---------------------------------------------------------------------------
echo "=== Starting monitoring stack ==="
export PROM_UID="$(id -u)" PROM_GID="$(id -g)"
start_monitoring

# ---------------------------------------------------------------------------
# 3. Start vLLM emulator via its compose file
# ---------------------------------------------------------------------------
echo "=== Starting vLLM emulator (8000) ==="
HF_HOME="${HF_HOME}" IMAGE_NAME="${AIC_IMAGE}" \
    VLLM_CPU_MODEL="${VLLM_CPU_MODEL:-facebook/opt-125m}" \
    docker compose -f "${EMULATOR_COMPOSE}" up -d

# ---------------------------------------------------------------------------
# 4. Wait for vLLM emulator to be healthy (up to 3 min)
# ---------------------------------------------------------------------------
echo "=== Waiting for vLLM emulator /health ==="
for i in $(seq 1 24); do
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        echo "vLLM emulator healthy after ~$((i * 8))s"
        break
    fi
    if [[ "${i}" -eq 24 ]]; then
        echo "ERROR: vLLM emulator did not become healthy in time" >&2
        docker logs aic-vllm-emulator >&2
        exit 1
    fi
    sleep 8
done

# ---------------------------------------------------------------------------
# 5. Issue 3 LLM prompts; assert non-empty completions
# ---------------------------------------------------------------------------
echo "=== Running LLM prompt smoke tests ==="
PROMPTS=(
    "The capital of France is"
    "In machine learning, a tensor is"
    "ROCm stands for"
)
for prompt in "${PROMPTS[@]}"; do
    response=$(curl -sf http://localhost:8000/v1/completions \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${VLLM_CPU_MODEL:-facebook/opt-125m}\",\"prompt\":\"${prompt}\",\"max_tokens\":20}" \
    )
    text=$(echo "${response}" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null || true)
    if [[ -z "${text}" ]]; then
        echo "ERROR: empty completion for prompt: ${prompt}" >&2
        echo "Response: ${response}" >&2
        exit 1
    fi
    echo "  OK: '${prompt}' -> '${text}'"
done

# ---------------------------------------------------------------------------
# 6. Wait one Prometheus scrape cycle, then health-check exporters
# ---------------------------------------------------------------------------
echo "=== Waiting for first scrape cycle ==="
sleep 12

echo "=== Exporter health check ==="
monitoring_healthcheck

# ---------------------------------------------------------------------------
# 7. Scrape all metric endpoints; generate HTML reference page
# ---------------------------------------------------------------------------
echo "=== Scraping metrics endpoints ==="
declare -A EXPORTER_PORTS=(
    [node_exporter]=9100
    [nvme_exporter]=9998
    [rdma_exporter]=9879
    [vllm]=8000
    [prometheus]=9090
)
METRICS_ARGS=()
for name in "${!EXPORTER_PORTS[@]}"; do
    port="${EXPORTER_PORTS[$name]}"
    outfile="/tmp/metrics_${name}.txt"
    curl -sf "http://localhost:${port}/metrics" > "${outfile}" 2>/dev/null || true
    if [[ -s "${outfile}" ]]; then
        METRICS_ARGS+=("--source" "${name}:${outfile}")
    fi
done

echo "=== Generating metrics HTML ==="
python3 "${WORKDIR}/monitoring/scripts/metrics_to_html.py" \
    "${METRICS_ARGS[@]}" \
    --sha "${SHA}" \
    --title "AIC Prometheus Metrics Reference" \
    > "${AIC_METRICS_PAGE_DIR}/index.html"

echo "=== Metrics page written to ${AIC_METRICS_PAGE_DIR}/index.html ==="
wc -l "${AIC_METRICS_PAGE_DIR}/index.html"

SRUN_BODY

chmod +x "${SRUN_SCRIPT}"

echo "=== Submitting CPU-only srun job ==="
srun \
    --controller="${SPUR_CONTROLLER_ADDR}" \
    ${AIC_SLURM_ACCOUNT:+--account="${AIC_SLURM_ACCOUNT}"} \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=16G \
    --time=00:30:00 \
    --partition=amd-spur \
    --gres= \
    bash "${SRUN_SCRIPT}" \
        "${WORKDIR}" \
        "${AIC_IMAGE}" \
        "${AIC_METRICS_PAGE_DIR}" \
        "${SHA}" \
        "${TARBALL_DIR}" \
        "${AIC_SMOKE_USE_REGISTRY}" \
        "${HF_HOME}"

echo "=== srun job completed ==="
REMOTE

# ---------------------------------------------------------------------------
# Copy metrics page artifact back to the runner working directory
# ---------------------------------------------------------------------------
AIC_REMOTE_METRICS_PAGE_DIR="${AIC_SHARED_NFS}/rocm-aic/metrics-pages-${SHORT}"

echo "=== Copying metrics page to runner ==="
mkdir -p metrics-page
scp -o ServerAliveInterval=30 \
    "${AIC_SPUR_HOST}:${AIC_REMOTE_METRICS_PAGE_DIR}/index.html" \
    metrics-page/index.html
ssh -o ServerAliveInterval=30 "${AIC_SPUR_HOST}" rm -rf "${AIC_REMOTE_METRICS_PAGE_DIR}"

echo "CPU monitoring smoke test passed for ${SHORT}"
