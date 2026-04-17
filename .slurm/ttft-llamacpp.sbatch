#!/bin/bash
#SBATCH --job-name=ttft-llamacpp
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --constraint="MARKHAM&(GFX90A|GFX942)"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=.slurm/logs/ttft-llamacpp-%j.log

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR}"
BENCH_DIR="${REPO_DIR}/benchmarks/ttft-llamacpp"

# ── venv ────────────────────────────────────────────────────────
VENV_DIR="/tmp/ttft-venv"
if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
fi
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
pip install -q openai matplotlib

# ── model ───────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-8B-Q4_K_M.gguf}"
if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "Downloading model ..."
    pip install -q huggingface-hub
    mkdir -p "$(dirname "${MODEL_PATH}")"
    hf download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf \
        --local-dir "$(dirname "${MODEL_PATH}")"
fi

# ── GPU arch ────────────────────────────────────────────────────
GPU_ARCH=$(rocm_agent_enumerator 2>/dev/null | grep -v '^gfx0' | sort -u | tail -1)
if [[ -z "${GPU_ARCH}" ]]; then
    echo "ERROR: could not detect GPU architecture"
    exit 1
fi
echo "Detected GPU arch: ${GPU_ARCH}"

# ── build llama.cpp (persistent + cached) ───────────────────────
LLAMACPP_TAG="${LLAMACPP_TAG:-b8799}"
BUILD_DIR="$HOME/.cache/llama.cpp-${LLAMACPP_TAG}-${GPU_ARCH}"

if [[ -x "${BUILD_DIR}/bin/llama-server" ]]; then
    echo "Using cached llama-server build: ${BUILD_DIR}"
else
    echo "Building llama.cpp ${LLAMACPP_TAG} for ${GPU_ARCH} ..."

    # clean up any failed previous build
    rm -rf "${BUILD_DIR}"

    SRC_DIR="/tmp/llama.cpp-${LLAMACPP_TAG}"
    rm -rf "${SRC_DIR}"
    git clone --depth 1 --branch "${LLAMACPP_TAG}" \
        https://github.com/ggml-org/llama.cpp.git "${SRC_DIR}"

    cmake -B "${BUILD_DIR}" -S "${SRC_DIR}" \
        -DGGML_HIP=ON \
        -DGGML_HIP_FA="${LLAMACPP_FA:-OFF}" \
        -DAMDGPU_TARGETS="${GPU_ARCH}" \
        -DCMAKE_BUILD_TYPE=Release

    cmake --build "${BUILD_DIR}" --target llama-server -j "$(nproc)"

    rm -rf "${SRC_DIR}"

    if [[ ! -x "${BUILD_DIR}/bin/llama-server" ]]; then
        echo "ERROR: build failed, no llama-server binary produced"
        exit 1
    fi
    echo "Build cached at: ${BUILD_DIR}"
fi
export PATH="${BUILD_DIR}/bin:${PATH}"

# ── run benchmark ───────────────────────────────────────────────
RESULTS="${REPO_DIR}/.slurm/logs/results-${SLURM_JOB_ID}.jsonl"

cd "${BENCH_DIR}"

python3 ttft_bench.py run \
    --model "${MODEL_PATH}" \
    --context-chars 400 4000 40000 \
    --repeats "${REPEATS:-10}" \
    --seed "${SEED:-42}" \
    --corpus-file "${BENCH_DIR}/corpus.txt" \
    --output "${RESULTS}"
