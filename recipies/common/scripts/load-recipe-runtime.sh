#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
	printf 'usage: %s <recipe-context> [recipe-root]\n' "$(basename "$0")" >&2
	exit 2
fi

context="$1"
recipe_root="${2:-}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
recipies_root="${repo_root}/recipies"

config_file="${RECIPE_RUNTIME_FILE:-${RECIPE_RUNTIME_CONFIG:-${RUNTIME_CONFIG_FILE:-}}}"
if [[ -z "${config_file}" ]]; then
	for candidate in "${recipe_root}/runtime.yaml" "${recipies_root}/runtime.yaml"; do
		if [[ -n "${candidate}" && -f "${candidate}" ]]; then
			config_file="${candidate}"
			break
		fi
	done
fi

if [[ -z "${config_file}" ]]; then
	exit 0
fi

if [[ ! -r "${config_file}" ]]; then
	printf 'error: recipe runtime YAML is not readable: %s\n' "${config_file}" >&2
	exit 1
fi

python3 - "${config_file}" "${context}" "${repo_root}" <<'PY'
import os
import re
import shlex
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "error: PyYAML is required to read recipe runtime YAML files"
    ) from exc


config_path = Path(sys.argv[1]).expanduser().resolve()
context = sys.argv[2]
repo_root = Path(sys.argv[3]).expanduser().resolve()


def normalize_key(key):
    return str(key).strip().lower().replace("-", "_")


def normalize_section(section):
    if not isinstance(section, dict):
        return {}
    return {normalize_key(key): value for key, value in section.items()}


def substitute_env(value):
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*(?::-[^}]*)?)\}")

    def replace(match):
        expr = match.group(1)
        if ":-" in expr:
            name, default = expr.split(":-", 1)
            return os.environ.get(name, default)
        return os.environ.get(expr, match.group(0))

    if isinstance(value, str):
        return pattern.sub(replace, value)
    if isinstance(value, list):
        return [substitute_env(item) for item in value]
    if isinstance(value, dict):
        return {key: substitute_env(item) for key, item in value.items()}
    return value


def scalar(value, separator=" "):
    if value is None:
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return separator.join(str(item) for item in value)
    return str(value)


def resolve_path(value):
    text = scalar(value)
    if text is None or text == "":
        return text
    expanded = Path(os.path.expanduser(text))
    if expanded.is_absolute():
        return str(expanded)
    return str(repo_root / expanded)


def put(env, name, value, *, separator=" ", path=False):
    if value is None:
        return
    if os.environ.get(name):
        return
    text = resolve_path(value) if path else scalar(value, separator)
    if text is None or text == "":
        return
    env[name] = text


with open(config_path, encoding="utf-8") as fh:
    config = yaml.safe_load(fh) or {}

if not isinstance(config, dict):
    raise SystemExit(f"error: runtime YAML must contain a mapping: {config_path}")

config = substitute_env(config)
exports = {}

for env_section in (
    config.get("env"),
    normalize_section(config.get("common", {})).get("env"),
):
    if env_section is None:
        continue
    if not isinstance(env_section, dict):
        raise SystemExit("error: runtime YAML env sections must be mappings")
    for key, value in env_section.items():
        put(exports, str(key), value)

context_keys = {
    "vllm-lmcache-hipfile": ("vllm_lmcache_hipfile", "vllm-lmcache-hipfile"),
    "vllm-lmcache-nixl": ("vllm_lmcache_nixl", "vllm-lmcache-nixl"),
    "vllm-atom-andy": ("vllm_atom_andy", "vllm-atom-andy"),
}
section = {}
for key in context_keys.get(context, (context,)):
    if key in config and isinstance(config[key], dict):
        section.update(normalize_section(config[key]))

host = normalize_section(config.get("host", {}))
paths = normalize_section(config.get("paths", {}))
server = normalize_section(config.get("server", {}))
lmcache = normalize_section(config.get("lmcache", {}))
slurm = normalize_section(config.get("slurm", {}))
benchmark = normalize_section(config.get("benchmark", {}))
secrets = normalize_section(config.get("secrets", {}))
nixl = normalize_section(config.get("nixl", {}))

for subsection_name, target in (
    ("host", host),
    ("paths", paths),
    ("server", server),
    ("lmcache", lmcache),
    ("slurm", slurm),
    ("benchmark", benchmark),
    ("secrets", secrets),
    ("nixl", nixl),
):
    sub = normalize_section(section.get(subsection_name, {}))
    target.update(sub)

put(exports, "IMAGE_NAME", host.get("image_name"))
put(exports, "GPU", host.get("gpu"))
put(exports, "CONTAINER_NAME", host.get("container_name"))
put(exports, "DATA", paths.get("data"), path=True)
put(exports, "DATA", host.get("data"), path=True)
put(exports, "LOG", paths.get("log"), path=True)
put(exports, "LOG", host.get("log"), path=True)
put(exports, "HF_HOME", paths.get("hf_home"), path=True)
put(exports, "HF_HOME", host.get("hf_home"), path=True)
put(exports, "HF_TOKEN_FILE", secrets.get("hf_token_file"), path=True)
put(exports, "CONTAINER_HF_HOME", host.get("container_hf_home"))
put(exports, "CONTAINER_DATA_DIR", host.get("container_data_dir"))
put(exports, "CONTAINER_LOG_DIR", host.get("container_log_dir"))
put(exports, "TZ", host.get("timezone"))
put(exports, "TZ", host.get("tz"))
put(exports, "ARGS", host.get("args"))
put(exports, "EXTRA_DOCKER_RUN_FLAGS", host.get("extra_docker_run_flags"))
put(exports, "ROCM_ARCH", host.get("rocm_arch"))
put(exports, "PYTORCH_ALLOC_CONF", server.get("pytorch_alloc_conf"))
put(exports, "VLLM_PYTORCH_ALLOC_CONF", server.get("pytorch_alloc_conf"))
put(exports, "VLLM_MODEL", server.get("model"))
put(exports, "VLLM_SERVER_DEV_MODE", server.get("dev_mode"))
put(exports, "VLLM_GPU_MEMORY_UTILIZATION", server.get("gpu_memory_utilization"))
put(exports, "VLLM_ENFORCE_EAGER", server.get("enforce_eager"))
put(exports, "VLLM_ENABLE_MFU_METRICS", server.get("enable_mfu_metrics"))
put(exports, "BOOK_SLUG", benchmark.get("book_slug"))
put(exports, "BOOK_SLUGS", benchmark.get("book_slugs"), separator=",")
put(exports, "BOOK_SLUG_FILE", benchmark.get("book_slug_file"), path=True)
put(exports, "BASE_SEED", benchmark.get("base_seed"))

if context == "vllm-lmcache-hipfile":
    prefix = "VLH"
    put(exports, "VLH_LMCACHE_IO", lmcache.get("io"))
    put(exports, "VLH_LMCACHE_LOG_LEVEL", lmcache.get("log_level"))
    put(exports, "VLH_LMCACHE_GDS_BUFFER_SIZE", lmcache.get("gds_buffer_size"))
    put(exports, "VLH_LMCACHE_ENABLE_CHUNK_STATISTICS", lmcache.get("enable_chunk_statistics"))
    put(exports, "VLH_LMCACHE_CHUNK_STATISTICS_STRATEGY", lmcache.get("chunk_statistics_strategy"))
    put(exports, "VLH_LMCACHE_ENABLE_KV_EVENTS", lmcache.get("enable_kv_events"))
    put(exports, "VLH_GPU_MEMORY_UTILIZATION", server.get("gpu_memory_utilization"))
    put(exports, "VLH_MAX_MODEL_LEN", server.get("max_model_len"))
    put(exports, "VLH_MAX_NUM_BATCHED_TOKENS", server.get("max_num_batched_tokens"))
    put(exports, "VLH_ENABLE_MFU_METRICS", server.get("enable_mfu_metrics"))
    put(exports, "VLH_ENFORCE_EAGER", server.get("enforce_eager"))
    put(exports, "VLH_PYTORCH_ALLOC_CONF", server.get("pytorch_alloc_conf"))
elif context == "vllm-lmcache-nixl":
    prefix = "VLN"
    put(exports, "VLN_LMCACHE_IO", lmcache.get("io"))
    put(exports, "VLN_LMCACHE_LOG_LEVEL", lmcache.get("log_level"))
    put(exports, "VLN_LMCACHE_ENABLE_CHUNK_STATISTICS", lmcache.get("enable_chunk_statistics"))
    put(exports, "VLN_LMCACHE_CHUNK_STATISTICS_STRATEGY", lmcache.get("chunk_statistics_strategy"))
    put(exports, "VLN_LMCACHE_ENABLE_KV_EVENTS", lmcache.get("enable_kv_events"))
    put(exports, "VLN_GPU_MEMORY_UTILIZATION", server.get("gpu_memory_utilization"))
    put(exports, "VLN_MAX_MODEL_LEN", server.get("max_model_len"))
    put(exports, "VLN_MAX_NUM_BATCHED_TOKENS", server.get("max_num_batched_tokens"))
    put(exports, "VLN_ENABLE_MFU_METRICS", server.get("enable_mfu_metrics"))
    put(exports, "VLN_ENFORCE_EAGER", server.get("enforce_eager"))
    put(exports, "VLN_PYTORCH_ALLOC_CONF", server.get("pytorch_alloc_conf"))
    put(exports, "VLN_NIXL_BUFFER_SIZE", nixl.get("buffer_size"))
    put(exports, "VLN_NIXL_BUFFER_DEVICE", nixl.get("buffer_device"))
    put(exports, "VLN_NIXL_POOL_SIZE", nixl.get("pool_size"))
    put(exports, "VLN_DOCKER_NOFILE", nixl.get("docker_nofile"))
    put(exports, "NIXL_PLUGIN_DIR", nixl.get("plugin_dir"), path=True)
    put(exports, "VLH_HIPFILE_STATS_LEVEL", nixl.get("hipfile_stats_level"))
elif context == "vllm-atom-andy":
    prefix = "VAA"
    put(exports, "VAA_LMCACHE_TIER", lmcache.get("tier"))
    put(exports, "VAA_LMCACHE_LOG_LEVEL", lmcache.get("log_level"))
    put(exports, "VAA_LMCACHE_MAX_LOCAL_CPU_SIZE", lmcache.get("max_local_cpu_size"))
    put(exports, "VAA_LMCACHE_ENABLE_CHUNK_STATISTICS", lmcache.get("enable_chunk_statistics"))
    put(exports, "VAA_LMCACHE_CHUNK_STATISTICS_STRATEGY", lmcache.get("chunk_statistics_strategy"))
    put(exports, "VAA_TENSOR_PARALLEL_SIZE", server.get("tensor_parallel_size"))
    put(exports, "VAA_GPU_MEMORY_UTILIZATION", server.get("gpu_memory_utilization"))
    put(exports, "VAA_MAX_MODEL_LEN", server.get("max_model_len"))
    put(exports, "VAA_MAX_NUM_BATCHED_TOKENS", server.get("max_num_batched_tokens"))
    put(exports, "VAA_ENABLE_MFU_METRICS", server.get("enable_mfu_metrics"))
    put(exports, "VAA_MODEL_PROFILE", server.get("model_profile"))
else:
    prefix = ""

if prefix in ("VLH", "VLN"):
    put(exports, f"{prefix}_SHARED_ROOT", slurm.get("shared_root"), path=True)
    put(exports, f"{prefix}_HF_HOME", slurm.get("hf_home"), path=True)
    put(exports, f"{prefix}_GUTENBERG_DATA_ROOT", slurm.get("gutenberg_data_root"), path=True)
    put(exports, f"{prefix}_NVME_BASE", slurm.get("nvme_base"), path=True)
    put(exports, f"{prefix}_NVME_AUTO_USE", slurm.get("nvme_auto_use"))
    put(exports, f"{prefix}_NVME_AUTO_DEVICE", slurm.get("nvme_auto_device"))
    put(exports, f"{prefix}_NVME_SCRATCH_FALLBACK", slurm.get("nvme_scratch_fallback"))
    put(exports, f"{prefix}_NVME_SCRATCH_ROOT", slurm.get("nvme_scratch_root"), path=True)
    put(exports, f"{prefix}_NVME_MIN_AVAIL_GB", slurm.get("nvme_min_avail_gb"))
    put(exports, f"{prefix}_NVME_USE_SHARED_DATA_DOCKER", slurm.get("nvme_use_shared_data_docker"))
    put(exports, f"{prefix}_NVME_MKFS", slurm.get("nvme_mkfs"))
    put(exports, f"{prefix}_NVME_MOUNT", slurm.get("nvme_mount"), path=True)
    put(exports, f"{prefix}_NVME_DEVICE", slurm.get("nvme_device"))
    put(exports, f"{prefix}_BENCHMARK", slurm.get("benchmark"))
    put(exports, f"{prefix}_BENCHMARK", benchmark.get("name"))
    put(exports, f"{prefix}_RUN_LONG_PARALLEL", benchmark.get("run_long_parallel"))
    put(exports, f"{prefix}_RUN_LONG_WORKERS", benchmark.get("workers"))
    put(exports, f"{prefix}_RUN_LONG_ITERATIONS", benchmark.get("iterations"))
    put(exports, f"{prefix}_RUN_LONG_BASE_SEED", benchmark.get("base_seed"))
    put(exports, f"{prefix}_RUN_LONG_MAX_TOKENS", benchmark.get("max_tokens"))
    put(exports, f"{prefix}_RUN_LONG_STAGGER_SEC", benchmark.get("stagger_sec"))
    put(exports, f"{prefix}_RUN_LONG_PROGRESS", benchmark.get("progress"))
    put(exports, f"{prefix}_VLLM_READY_TIMEOUT", slurm.get("vllm_ready_timeout"))
    put(exports, f"{prefix}_SKIP_BUILD", slurm.get("skip_build"))
    put(exports, f"{prefix}_NVME_BLK_BPFTRACE", slurm.get("nvme_blk_bpftrace"))
    put(exports, f"{prefix}_NVME_SMART_LOG", slurm.get("nvme_smart_log"))
    put(exports, f"{prefix}_VFS_BPFTRACE", slurm.get("vfs_bpftrace"))
    put(exports, f"{prefix}_SLURM_PARTITION", slurm.get("partition"))
    put(exports, f"{prefix}_SLURM_CONSTRAINT", slurm.get("constraint"))
    put(exports, f"{prefix}_SLURM_EXCLUDE", slurm.get("exclude"))
    put(exports, f"{prefix}_SLURM_NODELIST", slurm.get("nodelist"))
    put(exports, f"{prefix}_SLURM_MEM", slurm.get("mem"))
    put(exports, f"{prefix}_SLURM_CPUS", slurm.get("cpus"))
    put(exports, f"{prefix}_SLURM_TIME", slurm.get("time"))
    put(exports, f"{prefix}_SLURM_GRES", slurm.get("gres"))

for key in sorted(exports):
    print(f"export {key}={shlex.quote(exports[key])}")
PY
