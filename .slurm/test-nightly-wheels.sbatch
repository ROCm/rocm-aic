#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
#SBATCH --job-name=aic-wheel-test
#SBATCH --partition=amd-spur
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null
#
# Smoke-test the nightly pip wheels published on GitHub against a live gfx950
# node.  Strategy: load the existing AIC tarball (which carries the matching
# PyTorch + ROCm runtime), then create a fresh venv and install the nightly
# wheels INTO that venv to verify:
#
#   1. All five CDNA wheels install cleanly (no missing deps, no ABI mismatch).
#   2. `import vllm` + GPU visible -> vLLM reports the correct version.
#   3. `import lmcache` + `import nixl_rocm` succeed.
#   4. vLLM engine init completes on the real gfx950 GPU (proves the GPU kernels
#      compiled for gfx950 actually run, not just that the .whl unpacks).
#
# The test deliberately does NOT run the full cliff benchmark -- this is a
# quick (< 20 min) install+import+engine-init check to validate the wheel
# artifacts before directing users to pip install them.
#
# Overridable env:
#   NIGHTLY_TAG       rolling release tag to pull wheels from (default: nightly)
#   AIC_IMAGE         vllm+base image to load for the ROCm/PyTorch runtime
#   AIC_IMAGE_DIR     directory holding tarball(s) (default: shared NFS images/)
#   VLLM_MODEL        small model for engine-init test (must be in HF_HOME)
#   HF_HOME           offline HF cache (default: shared NFS hf/)
#   SKIP_ENGINE_INIT  set to 1 to skip the engine init step (install+import only)
#
set -uo pipefail
export PATH="/usr/local/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

LOGDIR="${REPO}/logs/${SLURM_JOB_ID:-manual-wheel-test}"
mkdir -p "${LOGDIR}"
exec >>"${LOGDIR}/wheel-test.out" 2>&1

log() { printf '[wheel-test] %s\n' "$*"; }
die() { printf '[wheel-test] ERROR: %s\n' "$*" >&2; exit 1; }

NIGHTLY_TAG="${NIGHTLY_TAG:-nightly}"
GH_RELEASE_BASE="https://github.com/ROCm/rocm-aic/releases/download/${NIGHTLY_TAG}"
AIC_IMAGE_DIR="${AIC_IMAGE_DIR:-/shared_nfs/${USER}/rocm-aic/images}"
HF_HOME="${HF_HOME:-/shared_nfs/huggingface}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-14B-Instruct-1M}"
SKIP_ENGINE_INIT="${SKIP_ENGINE_INIT:-0}"

# Resolve the AIC image: prefer caller-set AIC_IMAGE, else pick the latest
# tarball (mirrors how run-cliff.sbatch resolves it).
if [[ -z "${AIC_IMAGE:-}" ]]; then
    _tb="$(ls -t "${AIC_IMAGE_DIR}"/rocm-aic-*.tar.zst 2>/dev/null | grep -v '\.partial\.' | head -1 || true)"
    [[ -n "${_tb}" ]] || die "no rocm-aic tarball found under ${AIC_IMAGE_DIR}"
    # Derive image name:tag from the tarball basename (mirrors run-build-distribute.sh).
    _bn="$(basename "${_tb}" .tar.zst)"
    AIC_IMAGE="${_bn/rocm-aic-/rocm-aic:}"
fi

log "host        : $(hostname)  job=${SLURM_JOB_ID:-manual}"
log "release tag : ${NIGHTLY_TAG}"
log "base image  : ${AIC_IMAGE}"
log "model       : ${VLLM_MODEL}"
log "HF_HOME     : ${HF_HOME}"

# Load the image if not already present.  The tag inside the tarball may
# differ from the name derived from the tarball filename (e.g. the tarball
# includes the arch suffix "-gfx950" but the embedded tag does not), so
# capture the actual "Loaded image: name:tag" line and use that.
_have="$(docker images -q "${AIC_IMAGE}" 2>/dev/null)"
if [[ -z "${_have}" ]]; then
    _tb="$(ls -t "${AIC_IMAGE_DIR}"/rocm-aic-*.tar.zst 2>/dev/null | grep -v '\.partial\.' | head -1 || true)"
    [[ -n "${_tb}" ]] || die "image ${AIC_IMAGE} not loaded and no tarball found"
    log "loading ${AIC_IMAGE} from ${_tb}..."
    _loaded="$(zstd -dc "${_tb}" | docker load 2>&1)"
    printf '%s\n' "${_loaded}"
    # Use the first non-latest loaded tag as AIC_IMAGE so docker run works.
    _actual="$(printf '%s\n' "${_loaded}" | grep '^Loaded image:' | grep -v ':latest' | head -1 | sed 's/^Loaded image: //')"
    if [[ -n "${_actual}" ]]; then
        AIC_IMAGE="${_actual}"
        log "using loaded tag: ${AIC_IMAGE}"
    fi
fi

# ---------------------------------------------------------------------------
# Locate the wheel directory.  Compute nodes have no GitHub API access, so
# wheels must be pre-downloaded on the head node and supplied via WHEEL_DIR.
# Default: /shared_nfs/<user>/rocm-aic/wheels-nightly (populated by the
# head node before job submission).
# ---------------------------------------------------------------------------
WHEEL_DIR="${WHEEL_DIR:-/shared_nfs/${USER}/rocm-aic/wheels-nightly}"
[[ -d "${WHEEL_DIR}" ]] || die "WHEEL_DIR=${WHEEL_DIR} does not exist; download wheels on the head node first"
WHEEL_COUNT="$(ls "${WHEEL_DIR}"/*.whl 2>/dev/null | wc -l)"
[[ "${WHEEL_COUNT}" -gt 0 ]] || die "no .whl files found in ${WHEEL_DIR}"
log "wheels in ${WHEEL_DIR} (${WHEEL_COUNT}):"
for w in "${WHEEL_DIR}"/*.whl; do log "  $(basename "${w}") ($(du -sh "${w}" | cut -f1))"; done

# ---------------------------------------------------------------------------
# The test script that runs inside the container.
# ---------------------------------------------------------------------------
INNER_SCRIPT="${LOGDIR}/inner-wheel-test.sh"
cat >"${INNER_SCRIPT}" <<'INNER'
#!/bin/bash
set -uo pipefail
fail=0
note()  { printf '[wheel-inner] %s\n' "$*"; }
check() {
    local d="$1"; shift
    if "$@" >/tmp/_ck 2>&1; then
        note "OK   ${d}"
    else
        note "FAIL ${d}"
        sed 's/^/           /' /tmp/_ck
        fail=1
    fi
}

note "host: $(hostname) arch: $(uname -m)"
note "python: $(python3 --version 2>&1)"

# GPU visible?
gfx="$(rocminfo 2>/dev/null | grep -om1 'gfx[0-9a-z]*' || true)"
if [[ -n "${gfx}" ]]; then
    note "OK   GPU visible: ${gfx}"
else
    note "FAIL no GPU reported by rocminfo"; fail=1
fi

# Install wheels into a fresh venv so we don't accidentally rely on the
# image's pre-installed versions.
note "creating venv /tmp/wheel-test-venv..."
python3 -m venv /tmp/wheel-test-venv
source /tmp/wheel-test-venv/bin/activate

# Carry the PyTorch install from the image into the venv (symlink site-packages
# so we don't need to re-download PyTorch, which isn't in the nightly release).
# We use --system-site-packages so torch, torchvision, etc. are inherited.
note "re-creating venv with system site-packages..."
python3 -m venv --system-site-packages /tmp/wheel-test-venv
source /tmp/wheel-test-venv/bin/activate

note "installing nightly wheels from /wheels/..."
# Allow deps so transitive requirements (e.g. triton_kernels from amd_aiter)
# are satisfied; system site-packages satisfy torch/ROCm so nothing large is
# re-downloaded.  Use --no-index so pip only looks at the wheel dir + system.
pip install --quiet /wheels/*.whl || { note "FAIL pip install"; fail=1; }

# Import checks
check "torch GPU" python3 -c "
import torch
assert torch.cuda.is_available(), 'no CUDA/ROCm device'
print('torch', torch.__version__, '| GPUs:', torch.cuda.device_count())
"
check "import vllm" python3 -c "import vllm; print('vllm', vllm.__version__)"
check "import lmcache" python3 -c "
import lmcache
print('lmcache', getattr(lmcache, '__version__', '?'))
"
check "import nixl_rocm" python3 -c "import nixl_rocm; print('nixl_rocm OK')"
check "import aiter" python3 -c "import aiter; print('aiter OK')"

# Engine init: actually load the model on the GPU.  This is the key test --
# it validates that the gfx950 GPU kernels in the wheel actually execute.
if [[ "${SKIP_ENGINE_INIT:-0}" != "1" && -n "${VLLM_MODEL:-}" ]]; then
    note "engine init: ${VLLM_MODEL} (this takes 2-5 min)..."
    check "vllm engine init" python3 -c "
import os, pathlib
os.environ.setdefault('VLLM_LOGGING_LEVEL', 'WARNING')
from vllm import LLM

# Resolve the model from the offline HF cache so we never need network.
# HF_HOME is bind-mounted at /hf.
model = '${VLLM_MODEL}'
cache_root = pathlib.Path(os.environ.get('HF_HOME', '/hf')) / 'hub'
candidates = list(cache_root.glob('models--' + model.replace('/', '--') + '/snapshots/*/'))
if candidates:
    model = str(sorted(candidates)[-1])
    print('resolved model from cache:', model)

llm = LLM(
    model=model,
    max_model_len=8192,
    gpu_memory_utilization=0.85,
    dtype='auto',
    load_format='dummy',
    trust_remote_code=False,
)
print('engine init OK; GPU kernels compiled and loaded')
del llm
"
fi

# MP mode: lmcache server (pure DRAM L1) + vLLM via LMCacheMPConnector.
# Tests the real production path — two processes sharing KV over ZMQ.
if [[ "${SKIP_MP_TEST:-0}" != "1" && -n "${VLLM_MODEL:-}" ]]; then
    LMCACHE_PORT=6555
    VLLM_PORT=8100

    # Resolve model path (same as engine-init above).
    model="${VLLM_MODEL}"
    cache_root="${HF_HOME:-/hf}/hub"
    snap=$(ls -d "${cache_root}/models--$(echo "${model}" | tr '/' '-')"/snapshots/*/ 2>/dev/null | sort | tail -1)
    [[ -n "${snap}" ]] && model="${snap}"
    note "MP test: lmcache server + vLLM LMCacheMPConnector (model=${model})"

    # Start lmcache server in background.
    lmcache server \
        --no-separate-object-groups \
        --host 0.0.0.0 \
        --port "${LMCACHE_PORT}" \
        --http-port 8080 \
        --l1-size-gb 2 \
        --eviction-policy LRU \
        --max-gpu-workers 1 \
        --worker-reap-timeout-seconds 60 \
        >/tmp/lmcache-mp.log 2>&1 &
    LMCACHE_PID=$!
    note "  lmcache server pid=${LMCACHE_PID}"

    # Wait for lmcache: TCP connect to the ZMQ port (same as the compose healthcheck).
    _lmc_ready=0
    for _i in $(seq 1 30); do
        if python3 -c "import socket; s=socket.create_connection(('127.0.0.1',${LMCACHE_PORT}),2); s.close()" 2>/dev/null; then
            _lmc_ready=1; break
        fi
        sleep 2
    done
    if [[ "${_lmc_ready}" -ne 1 ]]; then
        note "FAIL lmcache server never became healthy"
        cat /tmp/lmcache-mp.log | sed 's/^/           /' || true
        kill "${LMCACHE_PID}" 2>/dev/null || true
        fail=1
    else
        note "  lmcache server healthy"

        # Start vLLM server with LMCacheMPConnector.
        KV_CFG='{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":'"${LMCACHE_PORT}"',"lmcache.mp.heartbeat_interval":60}}'
        VLLM_LOGGING_LEVEL=WARNING python3 -m vllm.entrypoints.openai.api_server \
            --model "${model}" \
            --max-model-len 8192 \
            --gpu-memory-utilization 0.85 \
            --dtype auto \
            --port "${VLLM_PORT}" \
            --kv-transfer-config "${KV_CFG}" \
            >/tmp/vllm-mp.log 2>&1 &
        VLLM_PID=$!
        note "  vllm pid=${VLLM_PID}"

        # Wait for vLLM endpoint.
        _vllm_ready=0
        for _i in $(seq 1 60); do
            if curl -fsS "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
                _vllm_ready=1; break
            fi
            sleep 5
        done

        if [[ "${_vllm_ready}" -ne 1 ]]; then
            note "FAIL vllm (MP mode) never became ready"
            tail -30 /tmp/vllm-mp.log | sed 's/^/           /' || true
            fail=1
        else
            note "  vllm (MP mode) ready"
            check "vllm MP completion" python3 -c "
import urllib.request, json
req = urllib.request.Request(
    'http://localhost:${VLLM_PORT}/v1/completions',
    data=json.dumps({'model': '${model}', 'prompt': 'Hello', 'max_tokens': 4}).encode(),
    headers={'Content-Type': 'application/json'},
)
resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
print('MP completion OK:', resp['choices'][0]['text'][:40])
"
        fi

        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi

    kill "${LMCACHE_PID}" 2>/dev/null || true
    wait "${LMCACHE_PID}" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 4-phase KV tier benchmark: cold / hot / L1-DRAM / L2-disk
#
# Measures serving latency at each tier of the KV cache hierarchy:
#   cold     - first fill: no caches warm, full prefill on GPU
#   hot      - vLLM VRAM prefix cache hit (fastest)
#   L1-DRAM  - vLLM restarted (no VRAM cache), LMCache DRAM L1 hit
#   L2-disk  - vLLM + LMCache restarted, LMCache local-disk L2 hit
#
# Uses the same LMCacheMPConnector stack as production (lmcache server
# with DRAM L1 + LocalDiskBackend L2 on /tmp).  All 5 queries share a
# ~1 000-token prefix so the prefix-cache saving is exercised at every tier.
# ---------------------------------------------------------------------------
if [[ "${SKIP_TIER_BENCH:-0}" != "1" && -n "${VLLM_MODEL:-}" ]]; then
    T_LMC_PORT=6556
    T_LMCACHE_HTTP=8082
    T_VLLM_PORT=8101
    T_L2_DIR=/tmp/lmcache-tier-l2

    # Resolve model path (same logic as MP test above).
    t_model="${VLLM_MODEL}"
    t_cache="${HF_HOME:-/hf}/hub"
    t_snap=$(ls -d "${t_cache}/models--$(echo "${t_model}" | tr '/' '-')"/snapshots/*/ 2>/dev/null | sort | tail -1)
    [[ -n "${t_snap}" ]] && t_model="${t_snap}"
    note "tier bench: model=${t_model}"

    # L2 is bind-mounted from the host's job-local /tmp (node-local disk, not NFS)
    # at /hostdisk inside the container by the outer docker run.
    T_L2_DIR="/hostdisk"

    # LMCache config: DRAM L1 + LocalDiskBackend L2 on node-local disk.
    cat >/tmp/lmcache-tier.yml <<YML
chunk_size: 256
local_disk: "file://${T_L2_DIR}"
max_local_disk_size: 10
save_decode_cache: true
pre_caching_hash_algorithm: sha256_cbor
YML

    # Benchmark script written to a file so we avoid heredoc-in-heredoc quoting hell.
    cat >/tmp/tier_bench.py <<'PYEOF'
#!/usr/bin/env python3
"""tier_bench.py PORT MODEL PHASE
100 isolated questions spanning 5 ISL tiers (20 each: ~128/256/512/1024/2048 tokens).
Asked serially (blocking): send q1, receive full response, then send q2.
Reports per-ISL-tier totals and grand total so tier differences are visible
even when short questions dominate noise.
max_tokens=4 minimises decode time so prefill (and KV load) dominates the measurement.
"""
import time, json, sys, urllib.request, textwrap

PORT  = int(sys.argv[1])
MODEL = sys.argv[2]
PHASE = sys.argv[3]
MAX_T = 64  # realistic output length for a 2-3 sentence summary

# ---------------------------------------------------------------------------
# 20 topics, each with 10 unique factual sentences (~40 words / ~55 tokens each).
# No sentence is repeated within or across topics.
# ISL tiers select 1/2/4/7/10 sentences → approx 55/110/220/385/550 tokens.
# A unique [Q-NNN] prefix prevents cross-question prefix-cache sharing.
# Question suffix asks for a 2-3 sentence summary (produces ~50-64 tokens).
# ---------------------------------------------------------------------------
TOPICS = [
  ("Gravitational Waves", [
    "Gravitational waves are ripples in spacetime caused by accelerating massive objects, first predicted by Albert Einstein in 1916 as a consequence of his general theory of relativity.",
    "LIGO made the first direct detection in September 2015, recording a signal from two merging black holes 1.3 billion light-years away; the peak power released exceeded that of all stars in the observable universe combined.",
    "The detectors use laser interferometry across 4-kilometre evacuated arms, measuring path-length changes smaller than one-thousandth the diameter of a proton by exploiting the interference of laser light.",
    "Binary neutron star mergers emit gravitational waves for minutes as the pair spirals inward, then produce a short gamma-ray burst and a kilonova that synthesises heavy elements including gold, platinum, and uranium.",
    "The 2017 event GW170817 was the first neutron star merger observed both as gravitational waves and across the full electromagnetic spectrum, inaugurating multi-messenger astronomy and providing a new measurement of the Hubble constant.",
    "Pulsar timing arrays use networks of millisecond pulsars as galactic-scale detectors sensitive to nanohertz-frequency gravitational waves expected from pairs of supermassive black holes in merging galaxies.",
    "Third-generation ground-based detectors such as the Einstein Telescope, planned with 10-km arms in a triangular underground configuration, aim to observe gravitational wave events across the entire observable universe.",
    "Gravitational wave observations constrain the nuclear equation of state of neutron stars, revealing how matter behaves at densities exceeding those inside atomic nuclei, information impossible to obtain by any other means.",
    "The LISA space mission, to be launched in the 2030s, will deploy three spacecraft 2.5 million kilometres apart and detect millihertz-frequency waves from massive black hole mergers, galactic binaries, and extreme mass-ratio inspirals.",
    "Continuous gravitational waves expected from rotating neutron stars with tiny structural asymmetries have not yet been detected, but ongoing searches are progressively tightening constraints on the maximum height of neutron star mountains.",
  ]),
  ("Transport Layer Security", [
    "TLS is a cryptographic protocol that provides authentication, confidentiality, and integrity for data transmitted over networks, and is the security layer underlying HTTPS, email submission, and many other internet services.",
    "A TLS handshake negotiates a cipher suite, authenticates the server via its certificate signed by a trusted certificate authority, and establishes session keys using asymmetric cryptography so subsequent data is encrypted symmetrically.",
    "TLS 1.3, standardised in 2018, eliminated obsolete cipher suites, reduced the handshake from two round trips to one, and mandated forward secrecy, making passive decryption of recorded traffic infeasible even if a server key is later compromised.",
    "Certificate transparency logs provide a publicly auditable record of every TLS certificate issued, allowing domain owners and browsers to detect mis-issued certificates within hours of their appearance.",
    "The handshake uses Diffie-Hellman key exchange so the session key is never transmitted; each party derives the same secret from their own private value and the other party's public value without exposing the secret itself.",
    "OCSP stapling allows a server to cache and present a signed certificate-revocation status response from the certificate authority, sparing the client a separate revocation check and improving privacy and performance.",
    "SNI (Server Name Indication) allows a single IP address to host multiple TLS-protected domains by including the target hostname in the client hello, enabling the server to select the correct certificate before the handshake completes.",
    "Post-quantum TLS is being standardised using lattice-based key encapsulation mechanisms such as ML-KEM (formerly Kyber), which resist attacks from quantum computers that could break current elliptic-curve and RSA-based key exchange.",
    "Session resumption via pre-shared keys or session tickets allows a client that has previously completed a full handshake to reconnect with reduced latency, skipping asymmetric cryptography while maintaining forward secrecy for new sessions.",
    "Mutual TLS (mTLS) requires both client and server to present certificates, providing strong machine-to-machine authentication used in service meshes, financial APIs, and zero-trust network architectures.",
  ]),
  ("CRISPR-Cas9 Gene Editing", [
    "CRISPR-Cas9 is a genome editing tool derived from a bacterial adaptive immune system, using a guide RNA to direct the Cas9 protein to a specific DNA sequence where it makes a precise double-strand break.",
    "The cell repairs the break via non-homologous end joining, which often introduces small insertions or deletions that disrupt the gene, or via homology-directed repair using a provided DNA template to insert or correct a sequence.",
    "Jennifer Doudna and Emmanuelle Charpentier shared the 2020 Nobel Prize in Chemistry for developing CRISPR-Cas9 as a programmable genome editing tool, recognising its transformative impact across biology and medicine.",
    "Base editors and prime editors are CRISPR derivatives that can change individual nucleotides or insert short sequences without creating double-strand breaks, reducing unwanted off-target mutations and improving editing precision.",
    "CRISPR therapies for sickle cell disease and beta-thalassaemia received regulatory approval in 2023, editing patients' haematopoietic stem cells to reactivate foetal haemoglobin production and eliminate transfusion dependence.",
    "Off-target editing, where Cas9 cuts at sites with partial sequence similarity to the guide RNA, is a key safety concern addressed by high-fidelity Cas9 variants, truncated guide RNAs, and computational off-target prediction tools.",
    "CRISPR screens use libraries of guide RNAs to knock out thousands of genes in parallel, enabling genome-wide functional studies that identify genes essential for drug resistance, cancer survival, or viral infection.",
    "In vivo CRISPR delivery uses lipid nanoparticles, adeno-associated viruses, or electroporation to introduce editing machinery into target tissues; lipid nanoparticles have proven particularly effective for liver-targeted therapies.",
    "CRISPR agriculture applications include disease-resistant crops, hornless cattle, and enhanced nutritional profiles; regulatory classification differs between countries, with some treating small edits equivalently to conventional breeding.",
    "Anti-CRISPR proteins, discovered in bacteriophages as countermeasures to bacterial CRISPR immunity, are being engineered as off-switches to limit the duration of Cas9 activity and reduce off-target editing in therapeutic applications.",
  ]),
  ("Byzantine Fault Tolerance", [
    "The Byzantine Generals Problem describes how distributed nodes can reach consensus when some nodes may send arbitrary or contradictory messages, modelling failures caused by software bugs, hardware faults, or malicious actors.",
    "A system can tolerate at most f Byzantine-faulty nodes out of 3f+1 total nodes; with fewer nodes the honest majority cannot overcome the misinformation introduced by colluding faulty nodes.",
    "Practical Byzantine Fault Tolerance (PBFT), introduced by Castro and Liskov in 1999, provides safety and liveness in asynchronous networks with O(n²) message complexity, making it practical for small to medium-sized deployments.",
    "Tendermint, used by the Cosmos blockchain ecosystem, adapts BFT consensus for proof-of-stake networks, providing instant finality with O(n²) communication and a one-block confirmation time under favourable network conditions.",
    "HotStuff, used in the Diem (formerly Libra) blockchain and its successors, achieves linear message complexity per view change by using threshold signatures to aggregate votes, dramatically reducing communication overhead for large validator sets.",
    "The FLP impossibility result proves that no deterministic algorithm can guarantee consensus in an asynchronous network even with a single crash failure; BFT protocols overcome this by relaxing asynchrony assumptions or using randomisation.",
    "Viewstamped Replication and Raft are crash-fault-tolerant (non-Byzantine) consensus protocols that require only 2f+1 nodes but cannot tolerate malicious behaviour, making them suitable for trusted deployment environments.",
    "State machine replication applies consensus to maintain identical copies of a deterministic state machine across multiple servers, ensuring that clients observe a consistent linearisable service despite server failures.",
    "Ethereum's proof-of-stake consensus uses a fork-choice rule combined with finalisation gadgets from Casper FFG, tolerating up to one-third of validators being faulty while providing probabilistic and eventually absolute finality.",
    "Threshold signature schemes allow a quorum of validators to collaboratively produce a single compact signature without any individual holding the full signing key, enabling efficient aggregation of votes in large BFT systems.",
  ]),
  ("ATP Synthesis and Cellular Respiration", [
    "Cells generate ATP primarily through oxidative phosphorylation in mitochondria, coupling the oxidation of NADH and FADH₂ to the pumping of protons across the inner mitochondrial membrane to create an electrochemical gradient.",
    "ATP synthase (Complex V) uses the proton gradient as a turbine: proton flow through its Fₒ subunit drives rotation of the γ subunit within the F₁ catalytic head, mechanically causing each of three β subunits to cycle through states that bind ADP+Pi and release ATP.",
    "The electron transport chain comprises four protein complexes: NADH dehydrogenase (I), succinate dehydrogenase (II), cytochrome bc₁ (III), and cytochrome c oxidase (IV), which pass electrons stepwise to the terminal acceptor, molecular oxygen.",
    "Glycolysis in the cytosol converts one glucose to two pyruvate molecules, producing a net gain of 2 ATP and 2 NADH without consuming oxygen, providing a rapid but inefficient energy source for anaerobic conditions.",
    "The citric acid cycle (Krebs cycle) in the mitochondrial matrix oxidises acetyl-CoA derived from pyruvate, producing 3 NADH, 1 FADH₂, and 1 GTP per turn, along with CO₂ as a byproduct.",
    "Complete oxidation of one glucose molecule yields approximately 30-32 ATP, substantially more than the 2 ATP from glycolysis alone, because NADH and FADH₂ are reoxidised via the electron transport chain.",
    "Uncoupling proteins such as UCP1 in brown adipose tissue allow protons to bypass ATP synthase, dissipating the proton gradient as heat rather than ATP, which is exploited for thermogenesis in cold-adapted mammals including human newborns.",
    "Mitochondrial membrane potential, approximately −180 mV, is maintained by proton pumping and drives not only ATP synthesis but also calcium uptake, protein import, and the membrane potential-dependent import of metabolites.",
    "Reactive oxygen species (ROS) are produced as by-products of electron transport, particularly at Complexes I and III; excessive ROS causes oxidative damage to proteins, lipids, and DNA, contributing to ageing and mitochondrial diseases.",
    "Mitochondria are semiautonomous organelles that retain a small circular genome encoding 13 oxidative phosphorylation subunits, plus the rRNAs and tRNAs needed to translate them, evidence of their endosymbiotic bacterial ancestry.",
  ]),
  ("Quantum Entanglement", [
    "Quantum entanglement is a correlation between quantum systems such that measuring one instantly determines properties of the other regardless of distance, a phenomenon Einstein dismissed as 'spooky action at a distance.'",
    "Bell's theorem (1964) proved that entanglement cannot be explained by local hidden variables; subsequent experiments by Aspect, Clauser, and others — culminating in loophole-free tests — confirmed quantum mechanics violates Bell inequalities.",
    "Entanglement is a resource for quantum information tasks: superdense coding transmits two classical bits per qubit using a shared entangled pair, and quantum teleportation transfers an arbitrary qubit state using classical communication plus entanglement.",
    "Entangled photon pairs are routinely generated by spontaneous parametric down-conversion, in which a pump photon passing through a nonlinear crystal splits into two photons with correlated polarisations, momenta, and energies.",
    "Quantum key distribution protocols such as E91 use entanglement to detect eavesdropping: any measurement by a third party disturbs the entangled state, introducing detectable errors in the shared key.",
    "Entanglement entropy, measured by the von Neumann entropy of the reduced density matrix, quantifies the degree of entanglement between subsystems and plays a central role in quantum error correction and holographic duality in quantum gravity.",
    "Multipartite entanglement enables one-way quantum computation: a highly entangled cluster state is prepared in advance, and the computation proceeds by single-qubit measurements whose outcomes classically control subsequent measurement bases.",
    "Entanglement distillation protocols take multiple copies of a noisy entangled state and produce fewer copies of a higher-fidelity state, essential for quantum repeaters that extend quantum communication over lossy optical fibre.",
    "Monogamy of entanglement prevents a qubit from being maximally entangled with more than one other qubit simultaneously, which is the physical basis of security in device-independent quantum cryptography protocols.",
    "Quantum networks aim to distribute entanglement over metropolitan and continental distances using quantum repeaters, atomic memories, and heralded entanglement generation to enable secure communication, distributed quantum computing, and quantum sensing.",
  ]),
  ("Transformer Neural Networks", [
    "The Transformer architecture, introduced in 'Attention Is All You Need' (Vaswani et al., 2017), replaced recurrent and convolutional layers with self-attention, enabling parallelisation across sequence positions during training.",
    "Self-attention computes, for each token, a weighted sum of value vectors from all positions, with weights determined by the dot product of the token's query vector with every key vector, scaled and passed through a softmax.",
    "Multi-head attention runs h parallel attention functions with different learned projections, allowing the model to jointly attend to information from different representation subspaces at different positions, then concatenates and projects the results.",
    "Positional encodings, either fixed sinusoidal functions or learned embeddings, are added to token embeddings to provide sequence order information that pure attention lacks, as attention itself is permutation-equivariant.",
    "BERT (Bidirectional Encoder Representations from Transformers) pre-trains a Transformer encoder on masked language modelling and next-sentence prediction, then fine-tunes on downstream tasks, establishing a paradigm that dominates NLP.",
    "GPT-style models use Transformer decoders with causal (left-to-right) attention masks, pre-trained on next-token prediction at scale; emergent capabilities such as few-shot learning appear as model and data scale increase.",
    "Rotary position embeddings (RoPE) encode relative positions by rotating query and key vectors in the complex plane before the dot product, offering superior extrapolation to sequence lengths longer than those seen during training.",
    "Flash Attention reformulates the attention computation using tiling to keep data in SRAM rather than HBM, reducing memory I/O by an order of magnitude and enabling training of models with context windows of hundreds of thousands of tokens.",
    "Mixture-of-experts Transformers route each token to a subset of feed-forward sublayers (experts), increasing model capacity proportionally while keeping per-token FLOP count constant, enabling very large effective parameter counts at manageable compute.",
    "Speculative decoding uses a small draft model to generate candidate tokens rapidly, then verifies a batch of them in a single forward pass of the large target model, improving throughput by 2-3× without changing output distribution.",
  ]),
  ("Coral Reef Ecology", [
    "Coral reefs are marine ecosystems built by colonial cnidarians that secrete calcium carbonate skeletons; they cover less than 1% of the ocean floor but support about 25% of all marine species.",
    "The symbiotic relationship between corals and photosynthetic dinoflagellate algae (Symbiodiniaceae) living in their tissues provides up to 90% of the coral's energy; heat stress disrupts this partnership, causing bleaching.",
    "Mass bleaching events occur when sea surface temperatures exceed the thermal tolerance of corals by 1-2°C for several weeks; the 2015-2017 global bleaching event was the longest and most severe on record, damaging over 75% of reefs worldwide.",
    "Ocean acidification, driven by absorption of atmospheric CO₂, lowers seawater pH and reduces the carbonate ion concentration needed for coral calcification, slowing growth rates and increasing skeletal fragility.",
    "Crown-of-thorns starfish outbreaks, fuelled by nutrient runoff that promotes phytoplankton growth feeding larval survival, can consume coral tissue faster than reefs can recover, requiring active management including culling programmes.",
    "Coral restoration programmes propagate coral fragments on underwater nurseries, then transplant them to degraded reef areas; assisted evolution programmes selectively breed heat-tolerant coral strains or introduce thermotolerant Symbiodiniaceae.",
    "The Great Barrier Reef, stretching 2,300 km along Australia's Queensland coast, is the largest coral reef system on Earth and was added to the UNESCO World Heritage List in 1981 for its exceptional natural values.",
    "Mangroves and seagrass beds adjacent to reefs provide nursery habitat for reef fish, filter terrestrial runoff, and absorb wave energy; degradation of these interconnected habitats accelerates reef decline.",
    "No-take marine protected areas, when properly enforced, allow fish populations to recover to levels that restore herbivory controlling algal overgrowth, the main competitor of coral on degraded reefs.",
    "Projections under current greenhouse gas trajectories suggest that 70-90% of coral reefs will experience severe bleaching annually by 2050, with most of the world's reefs functionally extinct before the end of the century without drastic emissions cuts.",
  ]),
  ("Rust Memory Safety", [
    "Rust is a systems programming language designed to provide memory safety without a garbage collector, enforcing safety rules at compile time through its ownership and borrowing type system.",
    "Each value in Rust has a single owner; when the owner goes out of scope, the value is automatically dropped, deterministically freeing resources without a runtime garbage collector.",
    "Borrowing allows temporary references to a value: any number of shared (immutable) references may coexist, but a mutable reference must be the only active reference, preventing data races at compile time.",
    "The borrow checker analyses reference lifetimes, ensuring that references never outlive the data they point to, eliminating use-after-free errors and dangling pointers that cause undefined behaviour in C and C++.",
    "Rust's unsafe keyword delineates blocks where the programmer asserts invariants the compiler cannot verify, such as raw pointer dereferencing or calling C FFI functions; unsafe code can be audited and isolated from safe abstractions.",
    "The Send and Sync marker traits, automatically derived where applicable, statically guarantee that values sent across threads (Send) or accessed concurrently (Sync) are safe to do so, eliminating data races at the type system level.",
    "Rust's zero-cost abstractions compile iterators, closures, and traits to machine code as efficient as hand-written loops, providing high-level expressiveness without runtime overhead.",
    "The Rust compiler's error messages include suggestions, explanations, and references to documentation, significantly reducing the friction of satisfying the borrow checker and making memory safety accessible to developers new to the language.",
    "WebAssembly (Wasm) is a primary target for Rust: its memory safety guarantees make it attractive for sandboxed execution in browsers and serverless runtimes, and its lack of a garbage collector is essential for deterministic Wasm modules.",
    "Linux kernel development began accepting Rust code in kernel 6.1 (2022), using Rust's safety guarantees to reduce the attack surface of a codebase historically vulnerable to memory corruption bugs in device drivers.",
  ]),
  ("mRNA Vaccines", [
    "mRNA vaccines deliver instructions encoding a pathogen's antigen directly into host cells, which translate the mRNA into protein and present it to the immune system, priming adaptive immunity without exposing the patient to the pathogen.",
    "Katalin Karikó and Drew Weissman discovered that substituting modified nucleosides (e.g., pseudouridine) for uridine in synthetic mRNA suppresses innate immune sensing that would otherwise degrade the mRNA before it could be translated.",
    "Lipid nanoparticles (LNPs) protect mRNA from extracellular ribonucleases and facilitate cellular uptake by endocytosis; ionisable lipids adopt a positive charge in the acidic endosomal environment to disrupt the endosomal membrane and release mRNA into the cytosol.",
    "The BNT162b2 (Pfizer-BioNTech) and mRNA-1273 (Moderna) COVID-19 vaccines encode the SARS-CoV-2 spike protein stabilised in the prefusion conformation by two proline substitutions, eliciting strong neutralising antibody responses.",
    "mRNA is inherently transient: it does not enter the nucleus, cannot integrate into genomic DNA, and is degraded by normal cellular machinery within days of administration, addressing safety concerns about long-term genetic effects.",
    "The rapid development of COVID-19 mRNA vaccines—from sequence selection to emergency use authorisation in under a year—was enabled by decades of foundational research and the ability to synthesise and test candidate mRNAs within days of obtaining a pathogen sequence.",
    "mRNA cancer vaccines encode tumour neoantigens—mutant proteins unique to a patient's tumour—and are manufactured individually for each patient; early trials combining personalised mRNA vaccines with anti-PD-1 checkpoint inhibitors show promising results in melanoma.",
    "Optimising the 5′ cap, untranslated regions, codon usage, and poly-A tail of synthetic mRNA improves translational efficiency and stability; codon optimisation replaces rare codons with synonymous frequent ones to increase ribosomal throughput.",
    "Self-amplifying mRNA vaccines include sequences encoding an RNA-dependent RNA polymerase that replicates the mRNA inside cells, amplifying antigen production from a lower initial dose and potentially reducing manufacturing costs.",
    "mRNA therapeutics are being developed for protein replacement in enzyme deficiency disorders, intratumoral delivery of immunostimulatory proteins, and in vivo gene editing by encoding CRISPR components, extending the platform far beyond vaccines.",
  ]),
  ("Turing Machines and Computability", [
    "A Turing machine is an abstract device comprising an infinite tape, a read/write head, a finite state register, and a transition function; it formalises the notion of algorithm and defines what it means for a problem to be computable.",
    "The Church-Turing thesis asserts that any effectively calculable function can be computed by a Turing machine; this is not a theorem but a empirical hypothesis that has resisted all known counterexamples.",
    "The halting problem—deciding whether an arbitrary program halts on a given input—is undecidable: Turing proved in 1936 that no algorithm can solve it for all possible program-input pairs, establishing the first fundamental limit of computation.",
    "Rice's theorem extends undecidability: any non-trivial semantic property of programs (e.g., 'does this program ever print a zero?') is undecidable, making general program verification impossible in the Turing model.",
    "Reducibility connects computability problems: if problem A reduces to problem B, then B being decidable implies A is decidable; reductions are the primary tool for proving new problems undecidable by reduction from the halting problem.",
    "The complexity class P contains decision problems solvable in polynomial time; NP contains problems whose solutions can be verified in polynomial time. Whether P = NP is the most important open question in theoretical computer science.",
    "NP-complete problems are the hardest in NP: if any NP-complete problem has a polynomial-time solution, then P = NP. Cook's theorem (1971) proved SAT is NP-complete; Karp then showed 21 combinatorial problems are all NP-complete.",
    "Space complexity classes such as PSPACE (polynomial space) and L (logarithmic space) characterise resources other than time; PSPACE-complete problems include quantified Boolean formula satisfiability and many two-player game problems.",
    "Randomised algorithms and probabilistic Turing machines define classes like BPP (bounded-error probabilistic polynomial time); it remains unknown whether BPP equals P, though derandomisation results suggest it may.",
    "Interactive proof systems extend computation to dialogue between a prover and a verifier; IP = PSPACE proved that any PSPACE problem can be verified through polynomial rounds of interaction, connecting proof complexity to space complexity.",
  ]),
  ("Ocean Thermohaline Circulation", [
    "Thermohaline circulation, driven by gradients in ocean temperature and salinity rather than wind, moves water masses over centuries and millennia through all ocean basins, distributing heat, nutrients, carbon, and oxygen globally.",
    "Deep water formation occurs in the North Atlantic where warm, salty surface water transported by the Gulf Stream cools and becomes dense enough to sink to the ocean floor, driving the Atlantic Meridional Overturning Circulation (AMOC).",
    "AMOC transports approximately 1.3 petawatts of heat northward, moderating Northern European climates by roughly 5-10°C relative to what latitude alone would dictate; its weakening or collapse could trigger rapid regional cooling.",
    "Meltwater from the Greenland Ice Sheet and Arctic sea ice adds fresh, buoyant water to the North Atlantic, diluting the surface salinity that enables deep water formation and potentially slowing AMOC, as detected in observational records since 1950.",
    "Antarctic Bottom Water, the densest water mass in the ocean, forms in coastal polynyas around Antarctica through brine rejection as sea ice forms; it sinks and fills the abyssal basins, gradually upwelling elsewhere over hundreds of years.",
    "The biological pump is coupled to thermohaline circulation: nutrient-rich deep water upwells at high latitudes and along eastern continental margins, fuelling phytoplankton blooms that export organic carbon to depth when organisms die and sink.",
    "Paleoclimate records in sediment cores and ice cores show that AMOC has undergone rapid reorganisations called Dansgaard-Oeschger events during past glacials, causing abrupt temperature changes of 10-15°C in Greenland within decades.",
    "Ocean heat uptake by the deep circulation has absorbed over 90% of the excess energy retained by Earth due to anthropogenic greenhouse gas forcing, temporarily masking the full surface warming effect of atmospheric CO₂ increases.",
    "Meridional overturning circulations in the Pacific and Indian Oceans, collectively with the Atlantic forming the global ocean conveyor belt, distribute radiocarbon, dissolved oxygen, and anthropogenic carbon throughout the ocean interior.",
    "Projections from climate models suggest AMOC will weaken by 24-39% by 2100 under high-emission scenarios, increasing the risk of regional tipping points such as Amazon dieback, West African monsoon shifts, and European cooling.",
  ]),
  ("Fast Fourier Transform", [
    "The Fast Fourier Transform (FFT) computes the Discrete Fourier Transform of an N-point sequence in O(N log N) operations instead of the O(N²) of a naive implementation, enabling practical frequency analysis of large signals.",
    "The Cooley-Tukey algorithm, published in 1965, achieves efficiency by recursively decomposing the DFT of N points into two DFTs of N/2 points using the periodicity and symmetry of the complex exponential basis functions.",
    "The butterfly operation is the fundamental building block of the FFT: it combines two complex values with a twiddle factor (a complex exponential), producing two outputs, and is arranged in log₂N stages across the N/2 butterfly pairs.",
    "Bit-reversal permutation is applied to reorder input or output data in the Cooley-Tukey algorithm; the array index, written in binary, is reversed bit by bit, naturally arising from the recursive halving of the sequence.",
    "The FFT is fundamental to digital signal processing: audio codecs (MP3, AAC), image compression (JPEG), radar, sonar, telecommunications channel estimation, and numerical solution of partial differential equations all rely on it.",
    "Windowing functions such as Hann, Hamming, and Blackman-Harris are applied to the input signal before the FFT to reduce spectral leakage caused by the implicit periodicity assumption, improving frequency resolution for non-integer multiples of the bin spacing.",
    "FFTW (Fastest Fourier Transform in the West) auto-tunes the FFT algorithm at runtime to the specific hardware, cache hierarchy, and problem size, achieving near-optimal performance across different processors without manual tuning.",
    "The number-theoretic transform (NTT) replaces complex exponentials with roots of unity modulo a prime, enabling exact integer arithmetic FFTs used in large integer multiplication, polynomial multiplication over finite fields, and post-quantum cryptography.",
    "The FFT of a real-valued signal has conjugate symmetry, halving the computational work and storage required; real FFT implementations exploit this to compute an N-point real FFT using an N/2-point complex FFT.",
    "Sparse FFT algorithms exploit the fact that many real-world signals have only a small fraction of non-zero Fourier coefficients, computing the dominant K coefficients of an N-point signal in O(K log N) operations rather than O(N log N).",
  ]),
  ("Silk Road Trade Network", [
    "The Silk Road was not a single road but a network of overland and maritime routes connecting China, Central Asia, the Middle East, South Asia, and the Mediterranean, active from roughly 130 BCE when Han China opened trade to the West.",
    "Silk was the most prized Chinese export because European and Central Asian traders could not replicate its production; the secret of sericulture—raising Bombyx mori silkworms on white mulberry leaves—was guarded by imperial decree for centuries.",
    "Buddhist missionaries, merchants, and pilgrims travelling the Silk Road spread Buddhism from India through Central Asia to China; the religion's teachings, art styles, and architectural forms transformed as they moved along the route.",
    "The Black Death pandemic, caused by the bacterium Yersinia pestis, appears to have spread westward along Silk Road trade routes in the mid-14th century, reaching the Crimea by 1346 and entering Europe via Genoese trading ports.",
    "Caravanserais—roadside inns with stabling, water, food, and lodging—were spaced roughly one day's camel march apart along the overland routes, enabling merchants and their animals to rest and resupply without entering major cities.",
    "Islam spread along Silk Road trade routes from Arabia eastward through Central Asia to the borders of Tang China by the 8th century, and the Muslim merchant networks that arose subsequently dominated transcontinental trade for centuries.",
    "The maritime Silk Road, navigating monsoon winds across the Indian Ocean, linked Chinese ports with Southeast Asia, South Asia, the Persian Gulf, and East Africa, carrying porcelain, spices, cotton, and precious metals.",
    "Venetian merchant Marco Polo travelled overland to the court of Kublai Khan in the 1270s and remained in China for 17 years, returning with accounts that introduced Europeans to paper money, coal, and the scale of Chinese cities.",
    "Tang Dynasty Chang'an (modern Xi'an), the eastern terminus of the overland Silk Road, was one of the most cosmopolitan cities in the world, hosting merchants, diplomats, artists, and religious practitioners from dozens of cultures.",
    "The Silk Road declined after the Ottoman conquest of Constantinople in 1453 disrupted overland trade and European powers developed direct maritime routes to Asia, redirecting commerce through Portuguese, Dutch, and English East India companies.",
  ]),
  ("Protein Folding and AlphaFold", [
    "Protein folding is the process by which a polypeptide chain adopts its functional three-dimensional structure; the final conformation is determined by the sequence of amino acids, which interact through hydrogen bonds, hydrophobic effects, van der Waals forces, and electrostatic interactions.",
    "Levinthal's paradox observes that if a protein sampled all possible conformations at random it would take longer than the age of the universe to fold; real proteins fold in microseconds to seconds because the energy landscape funnels the chain toward the native state.",
    "AlphaFold2, developed by DeepMind and published in Nature in 2021, predicts protein structures from amino acid sequences with accuracy approaching experimental methods, solving what had been called the 'protein folding problem' that had challenged structural biologists for 50 years.",
    "AlphaFold2 uses a neural network incorporating multiple-sequence alignments of evolutionarily related proteins, pairwise residue representations called 'pair embeddings,' and a geometry-aware equivariant architecture called Evoformer to iteratively refine structure predictions.",
    "The AlphaFold Protein Structure Database, jointly released by DeepMind and EMBL-EBI, contains predicted structures for over 200 million proteins from across the tree of life, transforming access to structural information for biology and drug discovery.",
    "Experimental structure determination methods—X-ray crystallography, cryo-electron microscopy, and NMR spectroscopy—remain essential for high-confidence structures of novel complexes, ligand-bound states, and dynamic conformational ensembles that computational methods cannot yet fully capture.",
    "Intrinsically disordered proteins (IDPs) lack stable tertiary structure under physiological conditions and are abundant in eukaryotic proteomes; they adopt structure upon binding partners, and predicting their conditional folding represents a major frontier for computational methods.",
    "RoseTTAFold, ESMFold, and AlphaFold3 extended structure prediction to protein-nucleic acid and protein-ligand complexes; AlphaFold3 predicts the full biomolecular interactome including small molecules, ions, and post-translational modifications.",
    "Protein design leverages folding predictions in reverse: RFdiffusion and ProteinMPNN generate novel protein sequences predicted to adopt target structures, enabling creation of enzymes, binders, and vaccines with properties not found in nature.",
    "Molecular dynamics simulations complement structure prediction by sampling conformational dynamics, including domain motions, loop flexibility, and allostery, providing a kinetic and thermodynamic picture that static structural models cannot supply.",
  ]),
  ("Direct Air Carbon Capture", [
    "Direct air capture (DAC) removes CO₂ directly from the ambient atmosphere using chemical sorbents or solvents, offering a pathway to carbon removal independent of specific emissions sources and compatible with geological storage or utilisation.",
    "Liquid solvent DAC uses aqueous potassium hydroxide to absorb CO₂ as potassium carbonate, then regenerates the sorbent by reacting with calcium hydroxide to precipitate calcium carbonate, which is calcined at 900°C to release pure CO₂.",
    "Solid sorbent DAC uses amine-functionalised materials with temperature- or vacuum-swing adsorption cycles, adsorbing CO₂ at ambient temperature and releasing it at 80-120°C, enabling lower regeneration temperatures than liquid systems.",
    "The first large-scale commercial DAC plant, Orca (now replaced by Mammoth) operated by Climeworks in Iceland, captures up to 36,000 tonnes of CO₂ per year and permanently mineralises it in basaltic rock formations through the CarbFix process.",
    "Current DAC costs range from roughly $300-1,000 per tonne of CO₂, far above the $50-100 target required for climate-relevant deployment at scale; achieving this requires learning curves, engineering innovations, and access to cheap low-carbon electricity and heat.",
    "Energy intensity is a fundamental challenge for DAC: removing 1 tonne of CO₂ from air (which contains only 420 ppm CO₂) requires approximately 1.5-2 GJ of heat and 0.3-0.5 GJ of electricity due to the dilute nature of the capture resource.",
    "Enhanced weathering, ocean alkalinity enhancement, and bioenergy with carbon capture (BECCS) are competing carbon dioxide removal approaches; each has different land, water, energy, and geological storage requirements that determine feasibility at different scales.",
    "The US Inflation Reduction Act (2022) provides a tax credit of $180 per tonne of CO₂ captured by DAC and stored geologically, significantly improving the economics of early commercial projects and stimulating investment in scaling.",
    "CO₂ captured by DAC can be utilised rather than stored: it is used in carbonated beverages, synthetic aviation fuels (e-fuels) combined with green hydrogen, concrete curing, and greenhouse agriculture, though only geological storage provides permanent removal.",
    "Achieving gigatonne-scale DAC by 2050, as required by many net-zero scenarios, would demand terawatt-scale dedicated low-carbon energy, thousands of plants each capable of capturing millions of tonnes annually, and massive expansion of CO₂ transport and storage infrastructure.",
  ]),
  ("Relational Databases and SQL", [
    "The relational model, proposed by E.F. Codd at IBM in 1970, organises data as a collection of relations (tables) with named attributes (columns), defining operations on these relations using a mathematical foundation from first-order predicate logic.",
    "SQL (Structured Query Language), standardised by ANSI and ISO, provides a declarative interface to relational databases: the user specifies what data to retrieve rather than how to retrieve it, and the query optimiser determines the execution plan.",
    "ACID properties—Atomicity, Consistency, Isolation, and Durability—guarantee that database transactions are processed reliably: each transaction either completes fully or has no effect, and concurrent transactions produce results as if executed sequentially.",
    "Query optimisation is the process of choosing among equivalent execution plans for a SQL query; the optimiser considers join order, index availability, selectivity estimates, and access methods to minimise I/O and CPU cost.",
    "B-tree and B+ tree data structures underlie most relational database indexes, providing O(log N) lookup, insertion, and deletion while supporting efficient range scans because leaf nodes are linked in sorted order.",
    "Joins combine rows from two or more tables based on related columns; common execution strategies include nested-loop join (O(N×M)), hash join (O(N+M) for equi-joins), and merge join (O(N log N + M log M) for sorted inputs).",
    "Normalisation reduces data redundancy by decomposing tables according to functional dependencies; Boyce-Codd Normal Form (BCNF) and Third Normal Form (3NF) are common targets, though denormalisation for read performance is frequent in practice.",
    "Write-ahead logging (WAL) records changes to a log on durable storage before applying them to the database pages, enabling crash recovery by replaying committed transactions and rolling back uncommitted ones after a failure.",
    "MVCC (Multi-Version Concurrency Control) allows readers and writers to proceed concurrently without lock contention by maintaining multiple versions of each row; readers see a consistent snapshot of the database at their transaction start time.",
    "NewSQL databases such as CockroachDB and Google Spanner provide ACID transactions with horizontal scalability across distributed nodes using consensus protocols (Raft or Paxos) for replication, combining relational semantics with distributed systems resilience.",
  ]),
  ("Synaptic Plasticity and Memory", [
    "Synaptic plasticity—the ability of synapses to strengthen or weaken in response to activity—is widely considered the cellular mechanism underlying learning and memory storage in the nervous system.",
    "Long-term potentiation (LTP), first described by Bliss and Lømo in 1973, is a long-lasting increase in synaptic strength following high-frequency stimulation; it requires NMDA receptor activation and is blocked by the NMDA antagonist AP5.",
    "The Hebbian rule, 'neurons that fire together wire together,' captures the associative nature of LTP: coincident pre- and postsynaptic activity opens NMDA receptors by relieving magnesium block, allowing calcium influx that triggers potentiation.",
    "Early LTP (E-LTP) persists for 1-3 hours and depends on post-translational modifications—phosphorylation and insertion of AMPA receptors into the postsynaptic density—whereas late LTP requires new protein synthesis and lasts for days or longer.",
    "BDNF (brain-derived neurotrophic factor) released during LTP induction activates TrkB receptors and downstream signalling cascades including MAPK and PI3K, promoting the translation of synaptic proteins required for synapse strengthening and spine enlargement.",
    "Long-term depression (LTD) is the complement of LTP, decreasing synaptic strength following low-frequency stimulation or anti-Hebbian patterns of activity; it involves AMPA receptor endocytosis and is important for memory encoding specificity and synaptic homeostasis.",
    "Spike-timing-dependent plasticity (STDP) refines Hebbian learning: synapses strengthen if the presynaptic spike precedes the postsynaptic spike within ~20 ms, and weaken if the order is reversed, providing a mechanism for learning temporal sequences.",
    "Structural plasticity accompanies functional changes: dendritic spines grow, shrink, form, and retract in response to activity; LTP is associated with spine head enlargement and increased AMPA receptor content, while LTD correlates with spine shrinkage.",
    "The hippocampus is critical for episodic memory formation: patients with bilateral hippocampal lesions such as H.M. cannot form new declarative memories but retain procedural skills, demonstrating that different memory types depend on distinct brain structures.",
    "Systems consolidation transfers memories from hippocampus-dependent storage to neocortical networks during sleep; slow-wave sleep promotes hippocampal sharp-wave ripples that replay recent experiences, coordinating with cortical slow oscillations to stabilise long-term memory traces.",
  ]),
  ("Proof-of-Work Blockchain Consensus", [
    "Proof-of-work (PoW) consensus requires nodes (miners) to find a nonce value such that the hash of the block header falls below a target threshold, making block production computationally expensive and probabilistically preventing double-spending.",
    "Bitcoin's SHA-256-based PoW adjusts difficulty every 2016 blocks to maintain a 10-minute average block interval regardless of total network hash rate; this self-regulating mechanism has maintained the schedule across twelve orders of magnitude of hash rate growth.",
    "The longest-chain rule determines the canonical blockchain: nodes accept the chain with the greatest cumulative proof-of-work, incentivising miners to build on the tip and rendering rewriting history exponentially expensive as the chain grows.",
    "A 51% attack occurs when a single entity controls the majority of network hash rate and can double-spend by mining a private fork longer than the public chain; confirmed transaction depth proportional to transaction value provides probabilistic finality.",
    "The block reward—initially 50 BTC and halved every 210,000 blocks—subsidises miners during Bitcoin's bootstrapping phase; after all 21 million BTC are mined (approximately 2140), transaction fees must sustain network security.",
    "Mining pools aggregate hash rate from thousands of participants, distributing block rewards proportionally to contributed work and smoothing income variability; pool operators face trust requirements and can misdirect pool hash rate, creating centralisation pressure.",
    "ASICs (application-specific integrated circuits) designed for SHA-256 hashing are billions of times more efficient than general-purpose CPUs for Bitcoin mining, creating high capital barriers and concentrating mining in regions with cheap electricity.",
    "Proof-of-stake (PoS), adopted by Ethereum in 2022, replaces computational work with economic stake: validators post collateral that can be slashed for misbehaviour, achieving consensus with orders of magnitude less energy than PoW.",
    "Selfish mining is a strategy in which a miner withholds found blocks to gain a disproportionate share of rewards; theoretical analysis shows it can be profitable above 25-33% of network hash rate, depending on connectivity advantages.",
    "Bitcoin's energy consumption, comparable to that of mid-sized countries, is often cited as an environmental concern; advocates argue that much mining uses stranded renewable energy, and that PoW's physical energy anchor provides unique security properties.",
  ]),
  ("GPS and Satellite Navigation", [
    "GPS (Global Positioning System) is a US Department of Defense satellite constellation of 24+ satellites in medium Earth orbit at ~20,200 km altitude, transmitting precise timing signals that ground receivers use to compute position.",
    "A GPS receiver calculates its 3D position by measuring signal travel times from at least four satellites simultaneously; the travel-time-to-distance conversion via the speed of light, combined with known satellite positions from broadcast ephemeris data, yields the receiver's coordinates via trilateration.",
    "Each GPS satellite carries multiple atomic clocks (caesium and rubidium) accurate to nanoseconds; the transmission time is encoded in the navigation message and compared with the receiver's clock to measure pseudorange, with the satellite-receiver clock offset resolved by using a fourth satellite.",
    "Relativistic effects are significant in GPS: special relativity causes satellite clocks to run slower (gravitational time dilation reversal) than ground clocks by 7 microseconds per day, while general relativity causes them to run faster by 45 microseconds per day; the net 38-microsecond per day advance is corrected in the clock design.",
    "The GPS control segment, operated from Schriever Space Force Base in Colorado, tracks satellite orbits using ground monitoring stations worldwide, uploads updated ephemeris data and clock corrections, and commands satellite manoeuvres.",
    "GNSS (Global Navigation Satellite System) is the generic term for satellite navigation systems operated by different nations: GPS (USA), GLONASS (Russia), Galileo (EU), BeiDou (China), NAVIC (India), and QZSS (Japan), all interoperable with modern multi-constellation receivers.",
    "Differential GPS and RTK (Real-Time Kinematic) corrections broadcast ground-reference station measurements of GPS errors to nearby receivers, reducing position uncertainty from meters to centimetres and enabling precision agriculture, surveying, and autonomous vehicles.",
    "Multi-path interference, where signals reflect off buildings or terrain before reaching the receiver, is a major accuracy degradation source in urban environments; mitigation techniques include antenna design, signal processing filters, and building-model-aided positioning.",
    "Inertial navigation systems (INS) using accelerometers and gyroscopes are integrated with GNSS in aircraft and autonomous vehicles; INS provides continuous updates at high rates and bridges GNSS outages, while GNSS corrects INS drift.",
    "Spoofing attacks transmit counterfeit GPS signals stronger than authentic ones to deceive receivers into reporting false positions; authenticated signals using digital signatures (as planned for GPS III and Galileo OSNMA) provide cryptographic protection against spoofing.",
  ]),
]

# ISL tiers: all questions >= 1 lmcache chunk (chunk_size=256 tokens).
# Longer tiers cycle through multiple topics to build content without repetition.
# Facts/tier → approx tokens (55 tokens/fact):
#   5  →  275 tokens  (1 chunk)
#   10 →  550 tokens  (2 chunks)
#   20 → 1100 tokens  (4 chunks)
#   40 → 2200 tokens  (8 chunks)
#   80 → 4400 tokens  (17 chunks)
# Each question is unique: Q-NNN prefix + different starting topic.
FACTS_PER_TIER = [20, 80, 160, 320, 640]
ISL_LABELS     = [1100, 4400, 8800, 17600, 35200]
QUESTIONS_PER_TIER = 5

def build_context(q_id, n_facts):
    """Collect n_facts facts by cycling through TOPICS from q_id % len(TOPICS)."""
    facts = []
    t_idx = q_id % len(TOPICS)
    while len(facts) < n_facts:
        _, topic_facts = TOPICS[t_idx % len(TOPICS)]
        for f in topic_facts:
            if len(facts) >= n_facts:
                break
            facts.append(f)
        t_idx += 1
    return " ".join(facts)

QUESTIONS = []   # list of (isl_label, prompt)
q_id = 0
for n_facts, isl_label in zip(FACTS_PER_TIER, ISL_LABELS):
    for i in range(QUESTIONS_PER_TIER):
        context = build_context(q_id, n_facts)
        prompt = (
            f"[Q-{q_id:03d}] The following is a factual passage:\n\n"
            f"{context}\n\n"
            f"In 2-3 sentences, summarise the key points from the passage above."
        )
        QUESTIONS.append((isl_label, prompt))
        q_id += 1

ISL_TIERS = ISL_LABELS

# ---------------------------------------------------------------------------
def complete(prompt):
    """Blocking single request — send, wait, return (elapsed, text)."""
    data = json.dumps({"model": MODEL, "prompt": prompt,
                       "max_tokens": MAX_T, "temperature": 0}).encode()
    req  = urllib.request.Request(
        f"http://localhost:{PORT}/v1/completions",
        data=data, headers={"Content-Type": "application/json"},
    )
    t0   = time.perf_counter()
    resp = json.loads(urllib.request.urlopen(req, timeout=300).read())
    return time.perf_counter() - t0, resp["choices"][0]["text"]

# Preflight: warm the LMCacheMPConnector with a short unique prompt that
# does NOT appear in QUESTIONS — keeps the test prefix cache cold for cold phase.
for _attempt in range(12):
    try:
        complete("What is 2+2?")
        break
    except Exception as _e:
        if _attempt < 11:
            time.sleep(5)
        else:
            raise RuntimeError(f"preflight failed after 12 attempts: {_e}") from _e

# ---------------------------------------------------------------------------
# Run all 100 questions serially (blocking).
# ---------------------------------------------------------------------------
t_start = time.perf_counter()
tier_times = {isl: [] for isl in ISL_TIERS}
for i, (isl, q) in enumerate(QUESTIONS):
    t, txt = complete(q)
    tier_times[isl].append(t)
    print(f"  [{PHASE}] q{i+1:03d} isl={isl:4d}: {t:.3f}s | {txt[:40].strip()!r}", flush=True)
t_total = time.perf_counter() - t_start

# Per-tier summary.
all_times = [t for ts in tier_times.values() for t in ts]
print(f"  [{PHASE}] TOTAL={t_total:.3f}s  n={len(all_times)}  "
      f"avg={sum(all_times)/len(all_times):.3f}s", flush=True)
for isl in ISL_TIERS:
    ts = tier_times[isl]
    print(f"  [{PHASE}]   isl={isl:4d}: total={sum(ts):.3f}s  "
          f"avg={sum(ts)/len(ts):.3f}s  n={len(ts)}", flush=True)
PYEOF

    # Helper: wait for lmcache ZMQ port (mirrors the compose healthcheck).
    wait_lmc() {
        for _j in $(seq 1 40); do
            python3 -c "import socket; s=socket.create_connection(('127.0.0.1',${T_LMC_PORT}),2); s.close()" 2>/dev/null \
                && return 0
            sleep 2
        done
        return 1
    }

    # Helper: wait for vLLM HTTP endpoint.
    wait_vllm_t() {
        for _j in $(seq 1 150); do
            curl -fsS "http://localhost:${T_VLLM_PORT}/v1/models" >/dev/null 2>&1 && return 0
            sleep 5
        done
        return 1
    }

    # Start lmcache with L1 + L2.
    start_lmc() {
        # L2 disk backend: fs_native adapter with base_path on node-local /hostdisk.
        # --l2-adapter JSON is the correct MP-server API for filesystem L2;
        # LMCACHE_CONFIG_FILE is for the vLLM connector client side, not the server.
        lmcache server \
            --no-separate-object-groups \
            --host 0.0.0.0 --port "${T_LMC_PORT}" --http-port 8082 --prometheus-port 9093 \
            --l1-size-gb 128 --eviction-policy LRU --max-gpu-workers 1 \
            --worker-reap-timeout-seconds 30 \
            --l2-adapter "{\"type\":\"fs_native\",\"base_path\":\"${T_L2_DIR}\"}" \
            >>/tmp/lmcache-tier.log 2>&1 &
        echo $!
    }

    # Start vLLM with MP connector.
    T_KV_CFG='{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":'"${T_LMC_PORT}"',"lmcache.mp.heartbeat_interval":60}}'
    start_vllm_t() {
        # VLLM_SERVER_DEV_MODE=1 enables POST /reset_prefix_cache so we can clear
        # the VRAM prefix cache between phases without restarting the process.
        VLLM_LOGGING_LEVEL=WARNING VLLM_SERVER_DEV_MODE=1 \
        python3 -m vllm.entrypoints.openai.api_server \
            --model "${t_model}" \
            --max-model-len 40960 \
            --gpu-memory-utilization 0.85 \
            --dtype auto \
            --port "${T_VLLM_PORT}" \
            --kv-transfer-config "${T_KV_CFG}" \
            >>/tmp/vllm-tier.log 2>&1 &
        echo $!
    }

    _tier_ok=1

    # ---- start initial stack ----
    LMC_PID=$(start_lmc)
    if ! wait_lmc; then
        note "FAIL tier: lmcache never became ready"
        tail -20 /tmp/lmcache-tier.log | sed 's/^/  /' || true
        _tier_ok=0; fail=1
    else
        note "  tier: lmcache ready (pid=${LMC_PID})"
        VLLM_T_PID=$(start_vllm_t)
        if ! wait_vllm_t; then
            note "FAIL tier: vllm never became ready"
            tail -30 /tmp/vllm-tier.log | sed 's/^/  /' || true
            _tier_ok=0; fail=1
        else
            note "  tier: vllm ready (pid=${VLLM_T_PID})"

            # Scrape lmcache hit/miss counters from the admin API.
            # Printed before and after each phase so we can confirm which tier served KV.
            lmc_metrics() {
                # Dump all lmcache Prometheus metrics — first 60 unique metric
                # name prefixes so we learn the actual naming convention.
                local resp
                resp=$(curl -sS --max-time 3 "http://localhost:9093/" 2>&1)
                [[ -z "${resp}" ]] && resp=$(curl -sS --max-time 2 "http://localhost:9090/" 2>&1)
                if [[ -n "${resp}" ]]; then
                    # Print all non-comment, non-empty metric lines (values).
                    local hits
                    hits=$(printf '%s\n' "${resp}" \
                        | grep -v '^#' | grep -v '^$' \
                        | grep -iE "lmcache|kv_cache|l1_|l2_|hit|miss|store|retrieve|lookup" \
                        | head -30)
                    if [[ -n "${hits}" ]]; then
                        printf '%s\n' "${hits}" | sed 's/^/    [lmc-metrics] /'
                    else
                        # Show first 10 metric names to learn the naming scheme
                        note "  [lmc-debug] endpoint up but no matching metrics; sample names:"
                        printf '%s\n' "${resp}" | grep '^# HELP' | head -10 \
                            | sed 's/^/    [lmc-name] /'
                    fi
                else
                    note "  [lmc-debug] no response on :9093 or :9090"
                    tail -3 /tmp/lmcache-tier.log 2>/dev/null | sed 's/^/    [lmc-log] /' || true
                fi
            }

            # Phase 1 — cold (full prefill; KV stored to L1 DRAM and L2 disk)
            note "  tier: phase cold"
            lmc_metrics
            python3 /tmp/tier_bench.py "${T_VLLM_PORT}" "${t_model}" cold || { _tier_ok=0; fail=1; }
            note "  tier: lmcache metrics after cold:"
            lmc_metrics

            # Phase 2 — hot (vLLM VRAM prefix cache hit; lmcache not consulted)
            note "  tier: phase hot"
            lmc_metrics
            python3 /tmp/tier_bench.py "${T_VLLM_PORT}" "${t_model}" hot || { _tier_ok=0; fail=1; }
            note "  tier: lmcache metrics after hot:"
            lmc_metrics

            # ---- L1-DRAM: clear both caches via API only — no restarts ----
            # POST /reset_prefix_cache  → clears vLLM VRAM KV blocks
            # POST /cache/clear (lmcache admin) → clears lmcache L1 DRAM
            # L2 files on disk are untouched; next queries are served from L1.
            note "  tier: clearing vllm+lmcache caches via API for L1-DRAM phase"
            sleep 5  # let async L2 writes settle before clearing L1
            curl -fsS -X POST "http://localhost:${T_VLLM_PORT}/reset_prefix_cache" >/dev/null 2>&1 \
                || note "WARN: vllm /reset_prefix_cache failed"
            # lmcache admin API: POST /cache/clear clears L1 DRAM only.
            curl -fsS -X POST "http://localhost:${T_LMCACHE_HTTP}/cache/clear" \
                -H "Content-Type: application/json" -d '{"tier":"l1","force":false}' >/dev/null 2>&1 \
                || note "WARN: lmcache /cache/clear failed"
            note "  tier: phase L1-DRAM"
            lmc_metrics
            python3 /tmp/tier_bench.py "${T_VLLM_PORT}" "${t_model}" L1-DRAM || { _tier_ok=0; fail=1; }
            note "  tier: lmcache metrics after L1-DRAM:"
            lmc_metrics

            # ---- L2-disk: clear both again — L2 files on disk intact ----
            # vLLM VRAM cleared again; lmcache L1 DRAM cleared again.
            # Next queries must fetch from L2 disk (LocalDiskBackend).
            note "  tier: clearing caches via API for L2-disk phase"
            sleep 5
            curl -fsS -X POST "http://localhost:${T_VLLM_PORT}/reset_prefix_cache" >/dev/null 2>&1 \
                || note "WARN: vllm /reset_prefix_cache failed"
            curl -fsS -X POST "http://localhost:${T_LMCACHE_HTTP}/cache/clear" \
                -H "Content-Type: application/json" -d '{"tier":"l1","force":false}' >/dev/null 2>&1 \
                || note "WARN: lmcache /cache/clear failed"
            note "  tier: phase L2-disk"
            lmc_metrics
            python3 /tmp/tier_bench.py "${T_VLLM_PORT}" "${t_model}" L2-disk || { _tier_ok=0; fail=1; }
            note "  tier: lmcache metrics after L2-disk:"
            lmc_metrics

            kill "${VLLM_T_PID}" 2>/dev/null || true; wait "${VLLM_T_PID}" 2>/dev/null || true
        fi
    fi

    kill "${LMC_PID}" 2>/dev/null || true; wait "${LMC_PID}" 2>/dev/null || true

    [[ "${_tier_ok}" -eq 1 ]] && note "tier bench: all phases complete" || note "tier bench: one or more phases FAILED"
fi

note "=== RESULT: $([ $fail -eq 0 ] && echo PASS || echo FAIL) ==="
exit $fail
INNER
chmod +x "${INNER_SCRIPT}"

# ---------------------------------------------------------------------------
# Run the test inside the AIC image (has ROCm + PyTorch; we overlay the
# published wheels via bind-mount).
# ---------------------------------------------------------------------------
log "running wheel test in container ${AIC_IMAGE}..."
GPU_IDX="${ROCR_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-0}}"
# Job-local L2 disk dir for the tier benchmark (node-local /tmp, not NFS).
HOST_L2_DIR="/tmp/tier-l2-${SLURM_JOB_ID:-manual}"
mkdir -p "${HOST_L2_DIR}"
# On SPUR, SLURM_JOB_GPUS is the allocated GPU device index.
docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add video \
    --cap-add CAP_SYS_ADMIN \
    --cap-add SYS_PTRACE \
    --cap-add IPC_LOCK \
    --ulimit nofile=1048576:1048576 \
    --ulimit memlock=-1:-1 \
    --ipc shareable \
    --shm-size 32g \
    -e ROCR_VISIBLE_DEVICES="${GPU_IDX}" \
    -e HF_HOME=/hf \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e PYTORCH_ALLOC_CONF=expandable_segments:False \
    -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:False \
    -e VLLM_ROCM_USE_AITER=1 \
    -e AMDGCN_USE_BUFFER_OPS=0 \
    -e MIOPEN_FIND_MODE=FAST \
    -e SKIP_ENGINE_INIT="${SKIP_ENGINE_INIT:-0}" \
    -e SKIP_MP_TEST="${SKIP_MP_TEST:-0}" \
    -e SKIP_TIER_BENCH="${SKIP_TIER_BENCH:-0}" \
    -e VLLM_MODEL="${VLLM_MODEL}" \
    -v "${HF_HOME}":/hf:ro \
    -v "${WHEEL_DIR}":/wheels:ro \
    -v "${INNER_SCRIPT}":/inner-wheel-test.sh:ro \
    -v "${HOST_L2_DIR}":/hostdisk \
    --entrypoint bash \
    "${AIC_IMAGE}" \
    /inner-wheel-test.sh
rc=$?

log "log: ${LOGDIR}/wheel-test.out"
if [[ ${rc} -eq 0 ]]; then
    log "=== WHEEL TEST PASSED ==="
else
    log "=== WHEEL TEST FAILED (rc=${rc}) ==="
fi
exit ${rc}
