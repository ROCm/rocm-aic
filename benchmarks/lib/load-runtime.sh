#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
	printf 'usage: %s <context> [benchmark-root]\n' "$(basename "$0")" >&2
	exit 2
fi

context="$1"
bench_root="${2:-}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
benchmarks_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${benchmarks_root}/.." && pwd)"

config_file="${RUNTIME_CONFIG_FILE:-${BENCHMARK_RUNTIME_FILE:-${BENCHMARK_RUNTIME_CONFIG:-}}}"
config_files=()
env_overrides_enabled=1
if [[ -f "${benchmarks_root}/runtime-defaults.yaml" ]]; then
	config_files+=("${benchmarks_root}/runtime-defaults.yaml")
fi
if [[ -n "${config_file}" ]]; then
	config_files+=("${config_file}")
	env_overrides_enabled=0
else
	for candidate in "${benchmarks_root}/runtime.yaml" "${bench_root}/runtime.yaml"; do
		if [[ -n "${candidate}" && -f "${candidate}" ]]; then
			config_files+=("${candidate}")
			env_overrides_enabled=0
		fi
	done
fi

if [[ "${#config_files[@]}" -eq 0 ]]; then
	exit 0
fi

for config_file in "${config_files[@]}"; do
	if [[ ! -r "${config_file}" ]]; then
		printf 'error: runtime YAML is not readable: %s\n' "${config_file}" >&2
		exit 1
	fi
done

python3 - "${context}" "${repo_root}" "${env_overrides_enabled}" "${config_files[@]}" <<'PY'
import os
import re
import shlex
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "error: PyYAML is required to read runtime YAML files"
    ) from exc


context = sys.argv[1]
repo_root = Path(sys.argv[2]).expanduser().resolve()
env_overrides_enabled = sys.argv[3] == "1"
config_paths = [Path(arg).expanduser().resolve() for arg in sys.argv[4:]]

ENV_OVERRIDE_ALLOWLIST = {
    "HF_TOKEN",
    "HF_TOKEN_FILE",
    "OPENAI_API_KEY",
}


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
    if os.environ.get(name) and (
        env_overrides_enabled or name in ENV_OVERRIDE_ALLOWLIST
    ):
        return
    text = resolve_path(value) if path else scalar(value, separator)
    if text is None or text == "":
        return
    env[name] = text


def deep_merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


config = {}
for config_path in config_paths:
    with open(config_path, encoding="utf-8") as fh:
        part = yaml.safe_load(fh) or {}
    if not isinstance(part, dict):
        raise SystemExit(
            f"error: runtime YAML must contain a mapping: {config_path}"
        )
    config = deep_merge(config, substitute_env(part))

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

server = normalize_section(config.get("server", {}))
paths = normalize_section(config.get("paths", {}))
put(exports, "BASE_URL", server.get("base_url"))
put(exports, "MODEL", server.get("model"))
put(exports, "BOOK_DATA_ROOT", paths.get("book_data_root"), path=True)
put(exports, "BOOK_DATA_ROOT", paths.get("gutenberg_data_root"), path=True)
put(exports, "AGENTX_DATA_ROOT", paths.get("agentx_data_root"), path=True)

context_keys = {
    "llm-prefill": ("llm_prefill", "llm-prefill"),
    "llm-agentx": ("llm_agentx", "llm-agentx"),
    "ttft-lmcache": ("ttft_lmcache", "ttft-lmcache"),
}
section = {}
for key in context_keys.get(context, (context,)):
    if key in config and isinstance(config[key], dict):
        section.update(normalize_section(config[key]))

if context == "llm-prefill":
    put(exports, "BASE_URL", section.get("base_url"))
    put(exports, "MODEL", section.get("model"))
    put(exports, "BOOK_DATA_ROOT", section.get("data_root"), path=True)
    put(exports, "BOOK_DATA_ROOT", section.get("book_data_root"), path=True)
    put(exports, "BOOK_DATA_ROOT", section.get("gutenberg_data_root"), path=True)
    put(exports, "ITERATIONS", section.get("iterations"))
    put(exports, "WORKERS", section.get("workers"))
    put(exports, "BASE_SEED", section.get("base_seed"))
    put(exports, "OUTPUT_DIR", section.get("output_dir"), path=True)
    put(exports, "STAGGER_SEC", section.get("stagger_sec"))
    put(exports, "PROGRESS", section.get("progress"))
    put(exports, "PROGRESS_WIDTH", section.get("progress_width"))
    put(exports, "BOOK_SLUG", section.get("book_slug"))
    put(exports, "BOOK_SLUGS", section.get("book_slugs"), separator=",")
    put(exports, "BOOK_SLUG_FILE", section.get("book_slug_file"), path=True)
    put(exports, "CONTEXT_FILE", section.get("context_file"), path=True)
    put(exports, "QUESTION", section.get("question"))
    put(exports, "RUN_LONG_SEED", section.get("seed"))
    put(exports, "RUN_LONG_COMBINE_CHUNKS", section.get("combine_chunks"))
    put(exports, "RUN_LONG_MAX_TOKENS", section.get("max_tokens"))
    put(exports, "RUN_LONG_STREAM_CHAT", section.get("stream_chat"), path=True)
elif context == "llm-agentx":
    put(exports, "BASE_URL", section.get("base_url"))
    put(exports, "MODEL", section.get("model"))
    put(exports, "ITERATIONS", section.get("iterations"))
    put(exports, "WORKERS", section.get("workers"))
    put(exports, "BASE_SEED", section.get("base_seed"))
    put(exports, "OUTPUT_DIR", section.get("output_dir"), path=True)
    put(exports, "STAGGER_SEC", section.get("stagger_sec"))
    put(exports, "AGENTX_DATA_ROOT", section.get("data_root"), path=True)
    put(exports, "AGENTX_SEED", section.get("seed"))
    put(exports, "AGENTX_MAX_REQUESTS", section.get("max_requests"))
    put(exports, "AGENTX_MAX_CONTEXT", section.get("max_context"))
    put(exports, "AGENTX_STRICT", section.get("strict"))
    put(exports, "AGENTX_HONOR_THINK_TIME", section.get("honor_think_time"))
    put(exports, "AGENTX_DRY_RUN", section.get("dry_run"))
    put(exports, "AGENTX_HF_HOME", section.get("hf_home"), path=True)
    put(exports, "MAX_TOKENS", section.get("max_tokens"))
    put(exports, "PYTHON", section.get("python"), path=True)
    put(exports, "RUN_AGENT_REPLAY", section.get("replay"), path=True)
elif context == "ttft-lmcache":
    put(exports, "LMCACHE_CONFIG_FILE", section.get("config_file"), path=True)
    put(exports, "HIT_RATES", section.get("hit_rates"), separator=" ")
    put(exports, "CONTEXT_TOKENS", section.get("context_tokens"))
    put(exports, "REPEATS", section.get("repeats"))
    put(exports, "SEED", section.get("seed"))
    put(exports, "CACHE_DIR", section.get("cache_dir"), path=True)
    put(exports, "RESULTS", section.get("results"), path=True)
    put(exports, "MODEL", section.get("model"))
    put(exports, "SERVER_URL", section.get("server_url"))
    put(exports, "VLLM_STARTUP_TIMEOUT", section.get("vllm_startup_timeout"))
    put(exports, "GPU_MEMORY_UTILIZATION", section.get("gpu_memory_utilization"))
    put(exports, "TENSOR_PARALLEL_SIZE", section.get("tensor_parallel_size"))
    put(exports, "LMCACHE_LOG_LEVEL", section.get("lmcache_log_level"))

for key in sorted(exports):
    print(f"export {key}={shlex.quote(exports[key])}")
PY
