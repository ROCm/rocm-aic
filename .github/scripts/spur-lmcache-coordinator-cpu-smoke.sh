#!/usr/bin/env bash
set -euo pipefail

# Runs on the self-hosted runner, SSHes to the SPUR head node, and submits a
# CPU-only srun job (no --gres=gpu). The compute-node job loads the AIC image
# produced for the requested SHA, starts only `lmcache coordinator`, and runs
# the key-directory smoke suite entirely through the coordinator's HTTP API.

SHA="${1:?usage: $0 <full-sha>}"
if [[ ! "${SHA}" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "ERROR: expected a full 40-character Git SHA, got: ${SHA}" >&2
    exit 2
fi
SHORT="${SHA:0:7}"
AIC_IMAGE="rocm-aic-ci-${SHORT}:latest"
AIC_SPUR_HOST="${AIC_SPUR_HOST:?AIC_SPUR_HOST must be set}"
AIC_SPUR_HOST="${AIC_SPUR_HOST//[$'\t\r\n ']}"
AIC_SHARED_NFS="${AIC_SHARED_NFS:?AIC_SHARED_NFS must be set}"
AIC_SPUR_CONTROLLER="${AIC_SPUR_CONTROLLER:?AIC_SPUR_CONTROLLER must be set}"
RUN_KEY="${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-0}"
REPO="https://github.com/ROCm/rocm-aic.git"

ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "${AIC_SPUR_HOST}" env \
    SHA="${SHA}" \
    REPO="${REPO}" \
    AIC_IMAGE="${AIC_IMAGE}" \
    AIC_SHARED_NFS="${AIC_SHARED_NFS}" \
    AIC_SPUR_CONTROLLER="${AIC_SPUR_CONTROLLER}" \
    SPUR_CONTROLLER_ADDR="${AIC_SPUR_CONTROLLER}" \
    RUN_KEY="${RUN_KEY}" \
    bash << 'REMOTE'
set -euo pipefail

SHORT="${SHA:0:7}"
SAFE_RUN_KEY="${RUN_KEY//[^a-zA-Z0-9_.-]/-}"
WORKDIR="$HOME/Projects/rocm-aic-coordinator-smoke.${SHORT}.${SAFE_RUN_KEY}"
TARBALL_DIR="${AIC_SHARED_NFS}/rocm-aic/images/aic-ci-${SHORT}"

cleanup() {
    status=$?
    set +e

    case "${WORKDIR}" in
        "$HOME"/Projects/rocm-aic-coordinator-smoke.*)
            rm -rf "${WORKDIR}"
            ;;
        *)
            echo "Refusing to clean unexpected WORKDIR: ${WORKDIR}" >&2
            ;;
    esac

    trap - EXIT
    exit "${status}"
}
trap cleanup EXIT

echo "=== Cloning ${REPO} at ${SHA} into ${WORKDIR} ==="
rm -rf "${WORKDIR}"
git clone --filter=blob:none --no-single-branch "${REPO}" "${WORKDIR}"
git -C "${WORKDIR}" checkout "${SHA}"

SRUN_SCRIPT="$(mktemp "${WORKDIR}/aic-coordinator-cpu-smoke-XXXXXX.sh")"

cat > "${SRUN_SCRIPT}" << 'SRUN_BODY'
#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${1}"
AIC_IMAGE="${2}"
TARBALL_DIR="${3}"
SHORT="${4}"
RUN_KEY="${5}"

safe_run_key="${RUN_KEY//[^a-zA-Z0-9_.-]/-}"
container_name="aic-lmcache-coordinator-smoke-${SHORT}-${SLURM_JOB_ID:-${safe_run_key}}"
container_started=0

cleanup_container() {
    status=$?
    if [[ "${container_started}" == "1" && "${status}" -ne 0 ]]; then
        echo "=== Coordinator container state ===" >&2
        docker inspect \
            --format 'status={{.State.Status}} exit={{.State.ExitCode}} error={{.State.Error}}' \
            "${container_name}" >&2 || true
        echo "=== Coordinator container logs ===" >&2
        docker logs "${container_name}" >&2 || true
    fi
    if [[ "${container_started}" == "1" ]]; then
        docker rm -f "${container_name}" >/dev/null 2>&1 || true
    fi
}
trap cleanup_container EXIT

echo "=== Loading ${AIC_IMAGE} from ${TARBALL_DIR} ==="
image_base="$(printf '%s' "${AIC_IMAGE}" | tr '/:' '--')"
tarball="$(
    find "${TARBALL_DIR}" -maxdepth 1 -type f \
        \( -name "${image_base}-*.tar.zst" \
           -o -name "${image_base}-*.tar.gz" \
           -o -name "${image_base}-*.tar" \) \
        -print -quit 2>/dev/null || true
)"
if [[ -z "${tarball}" ]]; then
    echo "ERROR: no tarball for ${AIC_IMAGE} in ${TARBALL_DIR}" >&2
    ls -la "${TARBALL_DIR}" >&2 || true
    exit 1
fi

case "${tarball}" in
    *.tar.zst) zstd -dc "${tarball}" | docker load ;;
    *.tar.gz)  gzip -dc "${tarball}" | docker load ;;
    *.tar)     docker load -i "${tarball}" ;;
    *)
        echo "ERROR: unsupported image tarball: ${tarball}" >&2
        exit 1
        ;;
esac

echo "=== Starting CPU-only LMCache coordinator ==="
docker run -d \
    --name "${container_name}" \
    --label aic.test=lmcache-coordinator-cpu-smoke \
    --publish 127.0.0.1::9300 \
    --entrypoint /usr/local/bin/lmcache \
    "${AIC_IMAGE}" \
    coordinator \
      --host 0.0.0.0 \
      --port 9300 \
      --chunk-size 256 \
    >/dev/null
container_started=1

devices="$(docker inspect --format '{{json .HostConfig.Devices}}' "${container_name}")"
if [[ "${devices}" != "[]" && "${devices}" != "null" ]]; then
    echo "ERROR: coordinator container has explicit device mappings: ${devices}" >&2
    exit 1
fi
if ! docker exec "${container_name}" \
    bash -c 'test ! -e /dev/kfd && test ! -e /dev/dri'; then
    echo "ERROR: GPU device nodes are visible inside the coordinator container" >&2
    exit 1
fi

published="$(docker port "${container_name}" 9300/tcp)"
coordinator_port="${published##*:}"
if [[ ! "${coordinator_port}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not resolve coordinator port from: ${published}" >&2
    exit 1
fi
coordinator_url="http://127.0.0.1:${coordinator_port}"

echo "=== Waiting for ${coordinator_url}/healthz ==="
healthy=0
for _ in $(seq 1 30); do
    if curl -fsS "${coordinator_url}/healthz" >/dev/null 2>&1; then
        healthy=1
        break
    fi
    if [[ "$(docker inspect --format '{{.State.Running}}' "${container_name}")" != "true" ]]; then
        echo "ERROR: coordinator container exited during startup" >&2
        exit 1
    fi
    sleep 2
done
if [[ "${healthy}" != "1" ]]; then
    echo "ERROR: coordinator did not become healthy within 60 seconds" >&2
    exit 1
fi

echo "=== Running coordinator key-directory HTTP smoke suite ==="
python3 "${WORKDIR}/.github/scripts/lmcache-coordinator-http-smoke.py" \
    --coordinator-url "${coordinator_url}"

echo "=== Coordinator CPU smoke test passed ==="
SRUN_BODY

chmod +x "${SRUN_SCRIPT}"

echo "=== Submitting CPU-only srun job ==="
srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=16G \
    --time=00:30:00 \
    --partition=amd-spur \
    bash "${SRUN_SCRIPT}" \
        "${WORKDIR}" \
        "${AIC_IMAGE}" \
        "${TARBALL_DIR}" \
        "${SHORT}" \
        "${SAFE_RUN_KEY}"

echo "=== srun job completed ==="
REMOTE

echo "LMCache coordinator CPU smoke test passed for ${SHORT}"
