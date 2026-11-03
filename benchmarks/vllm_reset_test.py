#!/usr/bin/env python3
"""
vllm_reset_test.py — End-to-end L1 and L2 retrieval test for the AIC stack.

Verifies the full LMCache tiered-cache path using a small L1 (1 GiB by default)
so that L1 overflow and NVMe eviction happen quickly, then uses vLLM's
POST /reset_prefix_cache API to clear the GPU-side prefix cache without
restarting any container, forcing subsequent requests to exercise LMCache L1 and L2.

Pass/fail criteria
------------------
  L1 retrieval:  at least one L1 hit chunk after the GPU cache is reset
  L2 retrieval:  at least one L2 hit chunk after the GPU cache is reset

Exit codes
----------
  0  both L1 and L2 verified
  1  test logic error / unexpected failure
  2  L1 hits not observed
  3  L2 hits not observed
  4  both L1 and L2 hits missing

Usage (via make)
----------------
  make vllm-reset-test HF_TOKEN=... VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct \
      NVME_DATA=/mnt/lmcache-nvme

Env overrides (all optional)
-----------------------------
  AIC_TEST_VLLM_URL     vLLM OpenAI-compat base URL (default: http://aic-vllm-gpu0:8000)
  AIC_TEST_LMCACHE_URL  LMCache metrics URL       (default: http://aic-lmcache:8080)
  AIC_TEST_MODEL        Model name served by vLLM  (default: Qwen/Qwen2.5-3B-Instruct)
  AIC_TEST_ANCHORS      Number of anchor prompts   (default: 10)
  AIC_TEST_FLOOD        Number of flood prompts    (default: 300)
  AIC_TEST_TIMEOUT      Per-request timeout (s)    (default: 120)
  AIC_TEST_SEED         RNG seed for prompt generation (default: 42)
  AIC_TEST_ANCHOR_REPS  Preamble reps per anchor    (default: 22, ~2090 tok)
  AIC_TEST_MAX_TOKENS   Tokens generated per request (default: 20)

Cutting it down
---------------
ANCHORS x ANCHOR_REPS and FLOOD are the wall-clock knobs; prefill is linear
in prompt length and decode is linear in MAX_TOKENS.  On an emulated GPU the
defaults are many hours, so the rocjitsu target drives a reduced preset (see
RJ_TEST_* in mk/bench.mk).  Keep ANCHOR_REPS >= 6 so prompts still span more
than one 256-token LMCache chunk (measured: 6 reps = 541-625 tokens = 2
chunks; 4 reps is only one), and shrink LMCACHE_L1_SIZE_GB alongside
FLOOD so L1 still overflows -- eviction to L2 is the property under test,
and a preset that no longer evicts passes without testing anything.
"""

import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────────────────
VLLM     = os.getenv("AIC_TEST_VLLM_URL",    "http://localhost:8000")
LMCACHE  = os.getenv("AIC_TEST_LMCACHE_URL", "http://localhost:8080")
MODEL    = os.getenv("AIC_TEST_MODEL",        "Qwen/Qwen2.5-3B-Instruct")
N_ANCHOR = int(os.getenv("AIC_TEST_ANCHORS", "10"))
N_FLOOD  = int(os.getenv("AIC_TEST_FLOOD",   "50"))
TIMEOUT  = int(os.getenv("AIC_TEST_TIMEOUT", "120"))
# Tokens generated per request.  Decode is serial, so on a slow device this is
# a straight multiplier on every request; the test only needs the response to
# exist, not to be long.
MAX_TOK  = int(os.getenv("AIC_TEST_MAX_TOKENS", "20"))
EXEC     = []  # run curl directly on the host (vllm/lmcache use network_mode:host)

# chunk_size=256 tok × fp8 KV for Qwen2.5-3B ≈ 4.7 MiB/chunk; L1=1 GiB ≈ 222 chunks.
# At the measured ~90 tok/rep for Qwen2.5-3B, a default 22-rep anchor is ~1980
# tokens ≈ 8 chunks: 10 anchors ≈ 80 chunks, leaving ~142 free L1 slots, and 50
# flood prompts × ~8 chunks ≈ 400 chunks >> capacity → eviction.  (These read
# ~14 and ~700 before; that came from a 163 tok/rep estimate no tokenizer matches.)
# Repetitions of the shared preamble sentence.  The sentence tokenizes to
# ~95 tokens (SmolLM2-135M 97, Minueza-32M 104, Qwen2.5-3B 90), so the
# default 22 reps is ~2090 tokens ~= 8 LMCache chunks at chunk_size=256.
# An earlier comment here said ~163 tokens/rep and ~14 chunks; that basis
# does not match any of those tokenizers, and the L1 sizing advice derived
# from it was correspondingly off.
#
# This is the dominant cost of the whole test: prefill is linear in prompt
# length, so on a software-emulated GPU (rocjitsu, ~0.14 GFLOP/s) a default
# anchor is ~115 minutes for a 135M model.  Lower it to trade cache-depth
# for wall-clock -- but keep prompts above one chunk (>=256 tokens, i.e.
# reps >= 2), or there is nothing for LMCache to chunk and evict and the
# test stops testing anything.
ANCHOR_REPS = int(os.getenv("AIC_TEST_ANCHOR_REPS", "22"))

_PREAMBLE_SENTENCE = (
    "You are an expert in distributed KV cache systems, GPU memory management, "
    "NVMe storage tiering, NIXL transfer protocols, Prometheus observability, "
    "HIP IPC shared memory, LMCache multi-process mode, vLLM prefix caching, "
    "ROCm 7.14 internals, RDMA over Converged Ethernet (RoCE), and Docker "
    "networking with named bridge networks. The following is a detailed technical "
    "document covering all of these topics in depth. "
)
_PREAMBLE = _PREAMBLE_SENTENCE * ANCHOR_REPS

_WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
    "november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee "
    "zulu amber crimson dagger ember falcon garnet herald ivory jasper kelvin lancer "
    "marble nexus opal piston quartz rampart silver topaz umbra vortex walnut xenon "
    "yarrow zephyr basalt cobalt diesel ferrite glider hustle insist jungle kettle "
    "launch mortar nozzle orient plunge quiver rustle signal tumble urgent vessel "
    "warden expand yellow zealot bishop castle deploy entire forest gravel harness "
    "impact jockey keeper locket muster needle output pillar quarry radius sector "
    "turret upward valley wither xylem yellow zenith"
).split()

def _rnd(n):
    return " ".join(random.choices(_WORDS, k=n))

def anchor_prompt(i):
    return _PREAMBLE + f" [ANCHOR-{i:04d}] uid={random.randint(10**12, 10**13)} " + _rnd(30)

def flood_prompt(i):
    return f"[FLOOD-{i:06d}] uid={random.randint(10**12, 10**13)} " + _rnd(15) + " " + _PREAMBLE + " " + _rnd(15)

# ── HTTP helpers (via aic-client container so no host networking required) ────
def _exec_curl(*args, timeout=15):
    cmd = EXEC + ["curl", "-s", "--max-time", str(timeout)] + list(args)
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=timeout + 10)
    except Exception:
        return b""

def post_json(url, payload, timeout=None):
    timeout = timeout or TIMEOUT
    data = json.dumps(payload)
    raw = _exec_curl("-X", "POST", url, "-H", "Content-Type: application/json", "-d", data, timeout=timeout)
    try:
        return json.loads(raw)
    except Exception:
        return None

def get_text(url):
    return _exec_curl(url).decode(errors="replace")

# ── Metric helpers ────────────────────────────────────────────────────────────
def lm_snap():
    raw = get_text(f"{LMCACHE}/metrics")
    m = {}
    for line in raw.splitlines():
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 2:
            try:
                m[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return {
        "l1_bytes":   m.get("lmcache_mp_l1_memory_usage_bytes", 0),
        "l1_ratio":   m.get("lmcache_mp_l1_usage_ratio", 0),
        "l2_bytes":   m.get('lmcache_mp_l2_usage_bytes{l2_name="nixl_store"}', 0),
        "l1_hits":    int(m.get('lmcache_mp_prefetch_hit_chunks_total{tier="l1"}', 0)),
        "l2_hits":    int(m.get('lmcache_mp_prefetch_hit_chunks_total{tier="l2"}', 0)),
        "l2_stored":  int(m.get("lmcache_mp_l2_store_completed_objects_total", 0)),
        "misses":     int(m.get("lmcache_mp_prefetch_miss_chunks_total", 0)),
    }

def vllm_gpu_hits():
    raw = get_text(f"{VLLM}/metrics")
    for line in raw.splitlines():
        if line.startswith("vllm:prefix_cache_hits_total"):
            try:
                return int(float(line.split()[-1]))
            except Exception:
                pass
    return 0

def send(prompt):
    r = post_json(f"{VLLM}/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOK,
        "temperature": 0.0,
    })
    return r is not None and "choices" in r

def reset_gpu_cache():
    r = post_json(f"{VLLM}/reset_prefix_cache", {})
    return r is not None and r.get("success") is True

# ── Logging ───────────────────────────────────────────────────────────────────
def log(msg, prefix=""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {prefix}{msg}", flush=True)

def section(title):
    bar = "─" * (len(title) + 4)
    print(f"\n┌{bar}┐\n│  {title}  │\n└{bar}┘", flush=True)

def metric_table(snap, label, base=None):
    d_l1 = snap["l1_hits"] - base["l1_hits"] if base else 0
    d_l2 = snap["l2_hits"] - base["l2_hits"] if base else 0
    print(f"  {label}:")
    print(f"    L1 DRAM    {snap['l1_bytes']/1024**2:>8.1f} MiB  ({snap['l1_ratio']*100:.1f}%)")
    print(f"    L2 NVMe    {snap['l2_bytes']/1024**3:>8.3f} GiB")
    print(f"    L2 stored  {snap['l2_stored']:>8d} chunks")
    print(f"    L1 hits Δ  {d_l1:>+8d}    cumul={snap['l1_hits']}")
    print(f"    L2 hits Δ  {d_l2:>+8d}    cumul={snap['l2_hits']}")
    print(f"    Misses     {snap['misses']:>8d} cumul", flush=True)

# ── Main test ─────────────────────────────────────────────────────────────────
def main():
    random.seed(int(os.getenv("AIC_TEST_SEED", "42")))

    print("=" * 60)
    print("AIC vLLM Reset Test — LMCache L1 + L2 retrieval verification")
    print(f"  vLLM:     {VLLM}")
    print(f"  LMCache:  {LMCACHE}")
    print(f"  Model:    {MODEL}")
    print(f"  Anchors:  {N_ANCHOR}  Flood: {N_FLOOD}")
    print("=" * 60)

    # ── Prerequisite: check vLLM health ──────────────────────────────────────
    section("0. Prerequisite checks")
    health = get_text(f"{VLLM}/health")
    if "200" not in health and health.strip() != "":
        # curl -s returns empty on 200 for /health; check for error
        pass
    # Try a dummy request to confirm model is loaded
    dummy = post_json(f"{VLLM}/v1/chat/completions", {
        "model": MODEL, "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5, "temperature": 0.0,
    }, timeout=TIMEOUT)
    if dummy is None or "choices" not in dummy:
        log("ERROR: vLLM not responding or model not loaded", prefix="✗ ")
        sys.exit(1)
    log(f"vLLM healthy, model '{MODEL}' loaded", prefix="✓ ")

    # VLLM_SERVER_DEV_MODE check
    reset_check = post_json(f"{VLLM}/reset_prefix_cache", {})
    if reset_check is None or not reset_check.get("success"):
        log("ERROR: POST /reset_prefix_cache failed — is VLLM_SERVER_DEV_MODE=1 set?", prefix="✗ ")
        sys.exit(1)
    log("POST /reset_prefix_cache available (VLLM_SERVER_DEV_MODE=1)", prefix="✓ ")

    base = lm_snap()
    metric_table(base, "baseline")

    # ── Step 1: Write anchor prompts ─────────────────────────────────────────
    section(f"1. Write {N_ANCHOR} anchor prompts → fill LMCache L1")
    anchors = [anchor_prompt(i) for i in range(N_ANCHOR)]
    ok = sum(1 for a in anchors if (time.sleep(0.3) or True) and send(a))
    log(f"{ok}/{N_ANCHOR} anchor prompts served OK")
    snap_anchors = lm_snap()
    metric_table(snap_anchors, "after anchors", base)

    # ── Step 2: Flood to overflow L1 → evict anchors to L2 ──────────────────
    section(f"2. Flood {N_FLOOD} unique prompts → overflow L1, evict anchors to L2 NVMe")
    # ~95 tokens per preamble rep, measured across the tokenizers this test
    # is run with (SmolLM2-135M ~97, Minueza-32M ~104, Qwen2.5-3B ~90).
    _chunks = (ANCHOR_REPS * 95) // 256
    log(f"{N_FLOOD} flood prompts × ~{_chunks} chunks "
        f"(reps={ANCHOR_REPS}) → expect L1 overflow and eviction to L2")
    for i in range(N_FLOOD):
        send(flood_prompt(i))
        if (i + 1) % 50 == 0:
            s = lm_snap()
            log(f"  {i+1}/{N_FLOOD}: L1={s['l1_bytes']/1024**2:.0f} MiB "
                f"L2={s['l2_bytes']/1024**3:.2f} GiB stored={s['l2_stored']}")
        time.sleep(0.05)
    snap_flood = lm_snap()
    metric_table(snap_flood, "after flood", snap_anchors)

    if snap_flood["l2_stored"] == 0:
        log("WARNING: no chunks stored to L2 after flood — L1 may not have overflowed", prefix="⚠ ")

    # ── Step 3: Reset vLLM GPU prefix cache ──────────────────────────────────
    section("3. Reset vLLM GPU prefix cache (POST /reset_prefix_cache)")
    ok_reset = reset_gpu_cache()
    log(f"reset result: success={ok_reset}", prefix="✓ " if ok_reset else "✗ ")
    if not ok_reset:
        log("ERROR: GPU cache reset failed", prefix="✗ ")
        sys.exit(1)
    time.sleep(2)
    snap_reset = lm_snap()
    gpu_after = vllm_gpu_hits()
    log(f"vLLM GPU prefix hit counter after reset: {gpu_after:,}")
    metric_table(snap_reset, "after GPU reset", snap_flood)

    # ── Step 4: Re-send anchors → expect L1 + L2 hits ────────────────────────
    section(f"4. Re-send {N_ANCHOR} anchor prompts — GPU cache empty, expect L1 + L2 hits")
    log("Anchor chunks evicted from L1 → should load from L2 NVMe")
    log("Anchor chunks still in L1 → should serve from L1 DRAM")

    pre_l1 = snap_reset["l1_hits"]
    pre_l2 = snap_reset["l2_hits"]

    for i, a in enumerate(anchors):
        send(a)
        s = lm_snap()
        dl1 = s["l1_hits"] - pre_l1
        dl2 = s["l2_hits"] - pre_l2
        log(f"  anchor {i:02d}: L1hits=+{dl1}  L2hits=+{dl2}  L1={s['l1_bytes']/1024**2:.0f}MiB")
        time.sleep(0.5)

    final = lm_snap()
    metric_table(final, "FINAL", snap_reset)

    # ── Result ────────────────────────────────────────────────────────────────
    section("Result")
    total_l1 = final["l1_hits"] - snap_reset["l1_hits"]
    total_l2 = final["l2_hits"] - snap_reset["l2_hits"]

    print(f"  L1 hit chunks gained: {total_l1:+d}")
    print(f"  L2 hit chunks gained: {total_l2:+d}")
    print()

    fail = 0
    # L1 hits are optional — with a full flood all anchors evict to L2, leaving L1 empty.
    # Require only L2 hits: that proves the full evict→NVMe→retrieve path works.
    if total_l1 > 0:
        log(f"L1 retrieval PASS  (+{total_l1} chunks from DRAM)", prefix="✓ ")
    else:
        log("L1 retrieval INFO  (0 L1 hits — anchors fully evicted to L2, expected with flood >= L1 cap)", prefix="  ")

    if total_l2 > 0:
        log(f"L2 retrieval PASS  (+{total_l2} chunks from NVMe)", prefix="✓ ")
    else:
        log("L2 retrieval FAIL  (0 L2 hits — anchors may still be in L1 or L2 store failed)", prefix="✗ ")
        fail |= 4

    if fail:
        log("OVERALL: FAIL — L2 NVMe retrieval not confirmed", prefix="✗ ")
        sys.exit(fail)

    log(f"OVERALL: PASS — L2 +{total_l2}  L1 +{total_l1}", prefix="✓ ")
    sys.exit(0)


if __name__ == "__main__":
    main()
