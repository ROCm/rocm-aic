#!/usr/bin/env python3
"""LMCache Simulation Tool - Main CLI entry point."""
import json
import os
import sys
import click
import importlib.util
from pathlib import Path
from typing import Any, Optional


def _load_sim_logging():
    """Load sibling ``sim_logging`` without package path hacks."""
    p = Path(__file__).resolve().parent / "sim_logging.py"
    spec = importlib.util.spec_from_file_location(
        "sim_logging", p
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load sim_logging")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sim_logging = _load_sim_logging()
sim_logging.configure_default_at_import()


def _import_module(name):
    """Import a module that has hyphens in its filename."""
    module_path = Path(__file__).parent / f"{name}.py"
    module_name = name.replace("-", "_")
    spec = importlib.util.spec_from_file_location(
        module_name, module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_config_generator = _import_module("config-generator")
_storage_manager = _import_module("storage-manager")
_workload_generator = _import_module("workload-generator")
_model_loader = _import_module("model-loader")
_tokenizer_interface = _import_module(
    "tokenizer-interface"
)
_download_conversations = _import_module(
    "download-conversations"
)

# Import engine-manager lazily so `--help` never pulls in torch/CUDA.
_engine_manager_mod = None


def _engine_manager_class():
    """LMCacheEngine wrapper; imports torch only when used."""
    global _engine_manager_mod
    if _engine_manager_mod is None:
        _engine_manager_mod = _import_module(
            "engine-manager"
        )
    return _engine_manager_mod.EngineManager


ConfigGenerator = _config_generator.ConfigGenerator
StorageManager = _storage_manager.StorageManager
WorkloadGenerator = _workload_generator.WorkloadGenerator
CHUNK_TOKENS_INDEX_NAME = (
    _workload_generator.CHUNK_TOKENS_INDEX_NAME
)
ModelLoader = _model_loader.ModelLoader
TokenizerWrapper = _tokenizer_interface.TokenizerWrapper


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

STORAGE_TYPES = [
    "filesystem",
    "local-disk",
    "block-device",
    "gds",
    "redis",
    "s3",
    "remote",
]


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version="1.0.0")
@click.option(
    "--verbose-lmcache",
    is_flag=True,
    help="Show LMCache library INFO logs on stderr "
         "(default: hidden). Same as env "
         "LMCACHE_SIM_VERBOSE_LMCACHE=1.",
)
def cli(verbose_lmcache: bool):
    """LMCache Simulation Tool."""
    # Logging is configured at import from argv/env; this
    # option exists for discovery (see --help).
    _ = verbose_lmcache


@cli.command(name="help", hidden=True)
@click.pass_context
def help_alias(ctx: click.Context):
    """Same as ``lmcache-sim --help``."""
    click.echo(ctx.parent.get_help())


@cli.command()
@click.option(
    "--storage-type",
    type=click.Choice(STORAGE_TYPES),
    required=True,
    help="Storage backend type",
)
@click.option(
    "--remote-url",
    help="LMCache remote_url (redis, s3, remote types)",
)
@click.option(
    "--s3-region",
    help="AWS region when --storage-type is s3",
)
@click.option(
    "--extra-config",
    "extra_config_path",
    type=click.Path(exists=True, dir_okay=False),
    help="YAML/JSON merged into generated LMCache config",
)
@click.option(
    "--fs-odirect",
    is_flag=True,
    help="Enable O_DIRECT on LMCache fs:// connector "
    "(filesystem / block-device only); merges "
    "extra_config after --extra-config",
)
@click.option(
    "--probe-remote",
    is_flag=True,
    help="TCP check for redis/lm-style remote_url "
    "before engine start",
)
@click.option(
    "--storage-path",
    help="Storage path (for filesystem or GDS)",
)
@click.option(
    "--block-device",
    help="Block device path (e.g., /dev/nvme0n1)",
)
@click.option(
    "--mount-point",
    help="Mount point for block device",
)
@click.option(
    "--create-fs",
    is_flag=True,
    help="Create filesystem on block device",
)
@click.option(
    "--config",
    default="configs/lmcache-config.yml",
    help="Config file path",
)
@click.option(
    "--chunk-size",
    type=int,
    default=256,
    help="KV cache chunk size",
)
@click.option(
    "--local-cpu",
    is_flag=True,
    help="Enable local CPU caching",
)
@click.option(
    "--max-local-cpu-size",
    type=float,
    default=5.0,
    help="Max local CPU cache size (GB)",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "xpu"]),
    default="cpu",
    help="Device to run on (default: cpu)",
)
@click.option(
    "--cufile-buffer-size",
    type=int,
    default=8192,
    help="CuFile buffer size (MiB) for GDS",
)
@click.option(
    "--model-name",
    default="lmcache_model",
    help="Model name for cache identification",
)
@click.option(
    "--worker-id",
    type=int,
    default=0,
    help="Worker ID",
)
@click.option(
    "--world-size",
    type=int,
    default=1,
    help="Total workers",
)
@click.option(
    "--kv-dtype",
    type=click.Choice(
        ["float16", "float32", "bfloat16", "uint8"]
    ),
    default="float16",
    help="KV cache data type",
)
@click.option(
    "--kv-shape",
    default="2,2,256,4,16",
    help="KV cache shape",
)
@click.option(
    "--use-mla",
    is_flag=True,
    help="Enable Multi-Level Attention",
)
@click.option(
    "--hf-model-name",
    help="Hugging Face model identifier",
)
@click.option(
    "--model-path",
    help="Local path to model",
)
@click.option(
    "--tokenizer-mode",
    type=click.Choice(["vocab-only", "text-to-tokens"]),
    default="vocab-only",
    help="Tokenizer mode",
)
@click.option(
    "--cache-dir",
    help="Directory to cache downloaded models",
)
@click.option(
    "--auto-kv-shape",
    is_flag=True,
    help="Auto-calculate KV shape from model config",
)
@click.option(
    "--local-only",
    is_flag=True,
    help="Only use local models, don't download",
)
@click.option(
    "--hf-token-file",
    help="Path to Hugging Face token file",
)
def start(
    storage_type: str,
    remote_url: Optional[str],
    s3_region: Optional[str],
    extra_config_path: Optional[str],
    fs_odirect: bool,
    probe_remote: bool,
    storage_path: Optional[str],
    block_device: Optional[str],
    mount_point: Optional[str],
    create_fs: bool,
    config: str,
    chunk_size: int,
    local_cpu: bool,
    max_local_cpu_size: float,
    device: str,
    cufile_buffer_size: int,
    model_name: str,
    worker_id: int,
    world_size: int,
    kv_dtype: str,
    kv_shape: str,
    use_mla: bool,
    hf_model_name: Optional[str],
    model_path: Optional[str],
    tokenizer_mode: str,
    cache_dir: Optional[str],
    auto_kv_shape: bool,
    local_only: bool,
    hf_token_file: Optional[str],
):
    """Start LMCache engine (in-process)."""
    try:
        storage_mgr = StorageManager()
        config_gen = ConfigGenerator()
        engine_mgr = _engine_manager_class()()

        final_kv_shape = kv_shape
        final_kv_dtype = kv_dtype

        if hf_model_name or model_path:
            final_kv_shape, final_kv_dtype = (
                _load_model_params(
                    hf_model_name,
                    model_path,
                    cache_dir,
                    local_only,
                    hf_token_file,
                    chunk_size,
                    kv_dtype,
                    auto_kv_shape,
                )
            )

        config_path = _setup_storage_and_config(
            storage_type,
            storage_path,
            block_device,
            mount_point,
            create_fs,
            config,
            chunk_size,
            local_cpu,
            max_local_cpu_size,
            cufile_buffer_size,
            remote_url,
            s3_region,
            extra_config_path,
            probe_remote,
            fs_odirect,
            storage_mgr,
            config_gen,
        )

        engine_mgr.create_engine(
            config_path=config_path,
            model_name=model_name,
            kv_shape=final_kv_shape,
            kv_dtype=final_kv_dtype,
            worker_id=worker_id,
            world_size=world_size,
            use_mla=use_mla,
        )
        click.echo("LMCache engine created and ready!")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--storage-type",
    type=click.Choice(STORAGE_TYPES),
    required=True,
    help="Storage backend type",
)
@click.option(
    "--remote-url",
    help="LMCache remote_url (redis, s3, remote types)",
)
@click.option(
    "--s3-region",
    help="AWS region when --storage-type is s3",
)
@click.option(
    "--extra-config",
    "extra_config_path",
    type=click.Path(exists=True, dir_okay=False),
    help="YAML/JSON merged into generated LMCache config",
)
@click.option(
    "--fs-odirect",
    is_flag=True,
    help="Enable O_DIRECT on LMCache fs:// connector "
    "(filesystem / block-device only); merges "
    "extra_config after --extra-config",
)
@click.option(
    "--probe-remote",
    is_flag=True,
    help="TCP check for redis/lm-style remote_url "
    "before engine start",
)
@click.option(
    "--storage-path",
    help="Storage path (for filesystem or GDS)",
)
@click.option(
    "--block-device",
    help="Block device path (e.g., /dev/nvme0n1)",
)
@click.option(
    "--mount-point",
    help="Mount point for block device",
)
@click.option(
    "--create-fs",
    is_flag=True,
    help="Create filesystem on block device",
)
@click.option(
    "--config",
    default="configs/lmcache-config.yml",
    help="Config file path",
)
@click.option(
    "--chunk-size",
    type=int,
    default=256,
    help="KV cache chunk size",
)
@click.option(
    "--local-cpu",
    is_flag=True,
    help="Enable local CPU caching",
)
@click.option(
    "--max-local-cpu-size",
    type=float,
    default=5.0,
    help="Max local CPU cache size (GB)",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "xpu"]),
    default="cpu",
    help="Device to run on (default: cpu)",
)
@click.option(
    "--cufile-buffer-size",
    type=int,
    default=8192,
    help="CuFile buffer size (MiB) for GDS",
)
@click.option(
    "--model-name",
    default="lmcache_model",
    help="Model name for cache identification",
)
@click.option(
    "--worker-id",
    type=int,
    default=0,
    help="Worker ID",
)
@click.option(
    "--world-size",
    type=int,
    default=1,
    help="Total workers",
)
@click.option(
    "--kv-dtype",
    type=click.Choice(
        ["float16", "float32", "bfloat16", "uint8"]
    ),
    default="float16",
    help="KV cache data type",
)
@click.option(
    "--kv-shape",
    default="2,2,256,4,16",
    help="KV cache shape",
)
@click.option(
    "--use-mla",
    is_flag=True,
    help="Enable Multi-Level Attention",
)
@click.option(
    "--hf-model-name",
    help="Hugging Face model identifier",
)
@click.option(
    "--model-path",
    help="Local path to model",
)
@click.option(
    "--tokenizer-mode",
    type=click.Choice(["vocab-only", "text-to-tokens"]),
    default="vocab-only",
    help="Tokenizer mode",
)
@click.option(
    "--cache-dir",
    help="Directory to cache downloaded models",
)
@click.option(
    "--auto-kv-shape",
    is_flag=True,
    help="Auto-calculate KV shape from model config",
)
@click.option(
    "--local-only",
    is_flag=True,
    help="Only use local models, don't download",
)
@click.option(
    "--hf-token-file",
    help="Path to Hugging Face token file",
)
@click.option(
    "--text-input",
    help="Text file or inline text for tokenization",
)
@click.option(
    "--conversation-file",
    help="JSON conversation file for conversation "
         "pattern",
)
@click.option(
    "--chunk-index",
    "chunk_index",
    type=click.Path(),
    default=None,
    help="JSONL token sidecar path; default "
         "<storage-path>/" + CHUNK_TOKENS_INDEX_NAME
         + " (patterns store-only / retrieve-only / lookup-only)",
)
@click.option(
    "--pattern",
    type=click.Choice(
        [
            "random",
            "steady-state",
            "conversation",
            "store-only",
            "retrieve-only",
            "lookup-only",
        ]
    ),
    default="random",
    show_default=True,
    help="Workload pattern",
)
@click.option(
    "--duration",
    type=float,
    help="Workload duration in seconds",
)
@click.option(
    "--num-operations",
    type=int,
    help="Number of workload operations",
)
@click.option(
    "--rate",
    type=float,
    help="Operations per second",
)
@click.option(
    "--key-range",
    type=int,
    default=10000,
    help="Key range for random / steady-state / "
         "store-only patterns",
)
@click.option(
    "--value-size",
    type=int,
    default=1024,
    help="Value size in bytes",
)
@click.option(
    "--read-ratio",
    type=float,
    default=0.8,
    help="Read ratio for steady-state pattern",
)
@click.option(
    "--output-format",
    type=click.Choice(["json", "text"]),
    default="text",
    help="Output format",
)
@click.option(
    "--per-op-store-log",
    type=click.Path(),
    default=None,
    help="JSONL: one object per successful store "
         "(timestamps, latency_ms, bytes_written)",
)
@click.option(
    "--per-op-log",
    type=click.Path(),
    default=None,
    help="JSONL: one object per operation (all op types; "
         "timestamps, latency_ms, success, cache_hit, "
         "kv_blocks, data_bytes)",
)
@click.option(
    "--cleanup",
    is_flag=True,
    help="Cleanup storage mounts after workload",
)
@click.option(
    "--concurrency",
    type=int,
    default=1,
    help="Concurrent conversation slots "
         "(conversation pattern only)",
)
@click.option(
    "--passes",
    type=int,
    default=1,
    help="Number of passes over the dataset",
)
@click.option(
    "--persist-cache",
    is_flag=True,
    help="Keep cache files between runs and "
         "report warm-cache state on startup",
)
@click.option(
    "--max-conversations",
    type=int,
    default=0,
    help="Max conversations to load "
         "(0 = all, conversation pattern only)",
)
@click.option(
    "--shuffle-conversations",
    is_flag=True,
    help="Randomize conversation order",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="RNG seed for reproducible shuffle",
)
def run(
    storage_type: str,
    remote_url: Optional[str],
    s3_region: Optional[str],
    extra_config_path: Optional[str],
    fs_odirect: bool,
    probe_remote: bool,
    storage_path: Optional[str],
    block_device: Optional[str],
    mount_point: Optional[str],
    create_fs: bool,
    config: str,
    chunk_size: int,
    local_cpu: bool,
    max_local_cpu_size: float,
    device: str,
    cufile_buffer_size: int,
    model_name: str,
    worker_id: int,
    world_size: int,
    kv_dtype: str,
    kv_shape: str,
    use_mla: bool,
    hf_model_name: Optional[str],
    model_path: Optional[str],
    tokenizer_mode: str,
    cache_dir: Optional[str],
    auto_kv_shape: bool,
    local_only: bool,
    hf_token_file: Optional[str],
    text_input: Optional[str],
    conversation_file: Optional[str],
    chunk_index: Optional[str],
    pattern: str,
    duration: Optional[float],
    num_operations: Optional[int],
    rate: Optional[float],
    key_range: int,
    value_size: int,
    read_ratio: float,
    output_format: str,
    per_op_store_log: Optional[str],
    per_op_log: Optional[str],
    cleanup: bool,
    concurrency: int,
    passes: int,
    persist_cache: bool,
    max_conversations: int,
    shuffle_conversations: bool,
    seed: Optional[int],
):
    """Create engine and run workload in one
    command."""
    if not duration and not num_operations:
        click.echo(
            "Error: --duration or --num-operations "
            "required",
            err=True,
        )
        sys.exit(1)

    if pattern in ("store-only", "retrieve-only", "lookup-only"):
        if not chunk_index and not storage_path:
            click.echo(
                "Error: patterns store-only / retrieve-only / "
                "lookup-only require --storage-path or "
                "--chunk-index",
                err=True,
            )
            sys.exit(1)

    engine_mgr = _engine_manager_class()()
    storage_mgr = StorageManager()

    try:
        config_gen = ConfigGenerator()

        final_kv_shape = kv_shape
        final_kv_dtype = kv_dtype
        tokenizer_wrapper = None

        if hf_model_name or model_path:
            final_kv_shape, final_kv_dtype, ml = (
                _load_model_params_full(
                    hf_model_name,
                    model_path,
                    cache_dir,
                    local_only,
                    hf_token_file,
                    chunk_size,
                    kv_dtype,
                    auto_kv_shape,
                    tokenizer_mode,
                )
            )
            if (
                ml
                and tokenizer_mode
                == "text-to-tokens"
            ):
                tokenizer_wrapper = TokenizerWrapper(
                    tokenizer=ml.get_tokenizer(),
                    mode=tokenizer_mode,
                )
                click.echo(
                    "Tokenizer enabled for "
                    "text-to-tokens mode"
                )

        config_path = _setup_storage_and_config(
            storage_type,
            storage_path,
            block_device,
            mount_point,
            create_fs,
            config,
            chunk_size,
            local_cpu,
            max_local_cpu_size,
            cufile_buffer_size,
            remote_url,
            s3_region,
            extra_config_path,
            probe_remote,
            fs_odirect,
            storage_mgr,
            config_gen,
        )

        if persist_cache and storage_path:
            _probe_cache(storage_path)

        engine_mgr.create_engine(
            config_path=config_path,
            model_name=model_name,
            kv_shape=final_kv_shape,
            kv_dtype=final_kv_dtype,
            worker_id=worker_id,
            world_size=world_size,
            use_mla=use_mla,
        )
        click.echo(
            "LMCache engine created and ready!"
        )

        workload_gen = WorkloadGenerator(
            engine=engine_mgr,
            tokenizer=tokenizer_wrapper,
        )

        pattern_kwargs = {
            "key_range": key_range,
            "value_size": value_size,
            "chunk_size": chunk_size,
        }
        if pattern == "steady-state":
            pattern_kwargs[
                "read_ratio"
            ] = read_ratio
        elif pattern == "conversation":
            if not conversation_file:
                click.echo(
                    "Error: --conversation-file "
                    "is required for conversation "
                    "pattern",
                    err=True,
                )
                sys.exit(1)
            pattern_kwargs[
                "conversation_file"
            ] = conversation_file
            pattern_kwargs[
                "concurrency"
            ] = concurrency
            pattern_kwargs[
                "max_conversations"
            ] = max_conversations
            pattern_kwargs[
                "shuffle_conversations"
            ] = shuffle_conversations
            pattern_kwargs["seed"] = seed

        if text_input and tokenizer_wrapper:
            pattern_kwargs[
                "text_input"
            ] = text_input

        click.echo("Starting workload...")

        chunk_index_dir = (
            storage_path if not chunk_index else None
        )
        chunk_index_file = chunk_index

        try:
            workload_gen.run_workload(
                pattern=pattern,
                duration=duration,
                num_operations=num_operations,
                rate=rate,
                output_format=output_format,
                passes=passes,
                chunk_index_dir=chunk_index_dir,
                chunk_index_file=chunk_index_file,
                per_op_store_log=per_op_store_log,
                per_op_log=per_op_log,
                **pattern_kwargs,
            )
        except KeyboardInterrupt:
            click.echo(
                "\nWorkload interrupted by user",
                err=True,
            )

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        import traceback

        click.echo(
            traceback.format_exc(), err=True
        )
        sys.exit(1)
    finally:
        engine_mgr.close()
        if cleanup:
            storage_mgr.cleanup()
            click.echo("Storage cleaned up")


@cli.command()
@click.option(
    "--chunk-index",
    "chunk_index",
    type=click.Path(),
    default=None,
    required=False,
    help="JSONL token sidecar (patterns "
         "store-only / retrieve-only / lookup-only)",
)
@click.option(
    "--pattern",
    type=click.Choice(
        [
            "random",
            "steady-state",
            "conversation",
            "store-only",
            "retrieve-only",
            "lookup-only",
        ]
    ),
    default="random",
    show_default=True,
    help="Workload pattern",
)
@click.option(
    "--duration",
    type=float,
    help="Duration in seconds",
)
@click.option(
    "--num-operations",
    type=int,
    help="Number of operations",
)
@click.option(
    "--rate",
    type=float,
    help="Operations per second",
)
@click.option(
    "--key-range",
    type=int,
    default=10000,
    help="Key range for random / steady-state / "
         "store-only patterns",
)
@click.option(
    "--value-size",
    type=int,
    default=1024,
    help="Value size in bytes",
)
@click.option(
    "--read-ratio",
    type=float,
    default=0.8,
    help="Read ratio for steady-state pattern",
)
@click.option(
    "--output-format",
    type=click.Choice(["json", "text"]),
    default="text",
    help="Output format",
)
@click.option(
    "--per-op-store-log",
    type=click.Path(),
    default=None,
    help="JSONL per successful store",
)
@click.option(
    "--per-op-log",
    type=click.Path(),
    default=None,
    help="JSONL: one line per operation (all types)",
)
@click.option(
    "--hf-model-name",
    help="Hugging Face model identifier",
)
@click.option(
    "--model-path",
    help="Local path to model",
)
@click.option(
    "--tokenizer-mode",
    type=click.Choice(
        ["vocab-only", "text-to-tokens"]
    ),
    default="vocab-only",
    help="Tokenizer mode",
)
@click.option(
    "--cache-dir",
    help="Directory to cache downloaded models",
)
@click.option(
    "--local-only",
    is_flag=True,
    help="Only use local models",
)
@click.option(
    "--hf-token-file",
    help="Path to Hugging Face token file",
)
@click.option(
    "--text-input",
    help="Text file or inline text",
)
@click.option(
    "--conversation-file",
    help="JSON conversation file",
)
@click.option(
    "--config",
    default="configs/lmcache-config.yml",
    help="Config file path",
)
@click.option(
    "--model-name",
    default="lmcache_model",
    help="Model name for cache identification",
)
@click.option(
    "--kv-dtype",
    type=click.Choice(
        ["float16", "float32",
         "bfloat16", "uint8"]
    ),
    default="float16",
    help="KV cache data type",
)
@click.option(
    "--kv-shape",
    default="2,2,256,4,16",
    help="KV cache shape",
)
@click.option(
    "--chunk-size",
    type=int,
    default=256,
    help="KV cache chunk size",
)
@click.option(
    "--auto-kv-shape",
    is_flag=True,
    help="Auto-calculate KV shape from model",
)
@click.option(
    "--worker-id",
    type=int,
    default=0,
    help="Worker ID",
)
@click.option(
    "--world-size",
    type=int,
    default=1,
    help="Total workers",
)
@click.option(
    "--use-mla",
    is_flag=True,
    help="Enable Multi-Level Attention",
)
@click.option(
    "--concurrency",
    type=int,
    default=1,
    help="Concurrent conversation slots",
)
@click.option(
    "--passes",
    type=int,
    default=1,
    help="Number of passes over the dataset",
)
@click.option(
    "--max-conversations",
    type=int,
    default=0,
    help="Max conversations to load (0 = all)",
)
@click.option(
    "--shuffle-conversations",
    is_flag=True,
    help="Randomize conversation order",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="RNG seed for reproducible shuffle",
)
def workload(
    chunk_index: Optional[str],
    pattern: str,
    duration: Optional[float],
    num_operations: Optional[int],
    rate: Optional[float],
    key_range: int,
    value_size: int,
    read_ratio: float,
    output_format: str,
    per_op_store_log: Optional[str],
    per_op_log: Optional[str],
    hf_model_name: Optional[str],
    model_path: Optional[str],
    tokenizer_mode: str,
    cache_dir: Optional[str],
    local_only: bool,
    hf_token_file: Optional[str],
    text_input: Optional[str],
    conversation_file: Optional[str],
    config: str,
    model_name: str,
    kv_dtype: str,
    kv_shape: str,
    chunk_size: int,
    auto_kv_shape: bool,
    worker_id: int,
    world_size: int,
    use_mla: bool,
    concurrency: int,
    passes: int,
    max_conversations: int,
    shuffle_conversations: bool,
    seed: Optional[int],
):
    """Run workload with a temporary in-process
    engine."""
    if not duration and not num_operations:
        click.echo(
            "Error: --duration or "
            "--num-operations required",
            err=True,
        )
        sys.exit(1)

    if pattern in ("store-only", "retrieve-only", "lookup-only"):
        if not chunk_index:
            click.echo(
                "Error: patterns store-only / retrieve-only / "
                "lookup-only require --chunk-index "
                "for the workload command",
                err=True,
            )
            sys.exit(1)

    engine_mgr = _engine_manager_class()()

    try:
        tokenizer_wrapper = None
        final_kv_shape = kv_shape
        final_kv_dtype = kv_dtype

        if hf_model_name or model_path:
            final_kv_shape, final_kv_dtype, ml = (
                _load_model_params_full(
                    hf_model_name,
                    model_path,
                    cache_dir,
                    local_only,
                    hf_token_file,
                    chunk_size,
                    kv_dtype,
                    auto_kv_shape,
                    tokenizer_mode,
                )
            )
            if (
                ml
                and tokenizer_mode
                == "text-to-tokens"
            ):
                tokenizer_wrapper = TokenizerWrapper(
                    tokenizer=ml.get_tokenizer(),
                    mode=tokenizer_mode,
                )

        engine_mgr.create_engine(
            config_path=config,
            model_name=model_name,
            kv_shape=final_kv_shape,
            kv_dtype=final_kv_dtype,
            worker_id=worker_id,
            world_size=world_size,
            use_mla=use_mla,
        )

        workload_gen = WorkloadGenerator(
            engine=engine_mgr,
            tokenizer=tokenizer_wrapper,
        )

        pattern_kwargs = {
            "key_range": key_range,
            "value_size": value_size,
            "chunk_size": chunk_size,
        }
        if pattern == "steady-state":
            pattern_kwargs[
                "read_ratio"
            ] = read_ratio
        elif pattern == "conversation":
            if not conversation_file:
                click.echo(
                    "Error: --conversation-file "
                    "is required for conversation "
                    "pattern",
                    err=True,
                )
                sys.exit(1)
            pattern_kwargs[
                "conversation_file"
            ] = conversation_file
            pattern_kwargs[
                "concurrency"
            ] = concurrency
            pattern_kwargs[
                "max_conversations"
            ] = max_conversations
            pattern_kwargs[
                "shuffle_conversations"
            ] = shuffle_conversations
            pattern_kwargs["seed"] = seed

        if text_input and tokenizer_wrapper:
            pattern_kwargs[
                "text_input"
            ] = text_input

        workload_gen.run_workload(
            pattern=pattern,
            duration=duration,
            num_operations=num_operations,
            rate=rate,
            output_format=output_format,
            passes=passes,
            chunk_index_dir=None,
            chunk_index_file=chunk_index,
            per_op_store_log=per_op_store_log,
            per_op_log=per_op_log,
            **pattern_kwargs,
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        engine_mgr.close()


@cli.command("verify")
@click.option(
    "--storage-type",
    type=click.Choice(STORAGE_TYPES),
    required=True,
    help="Storage backend type",
)
@click.option(
    "--remote-url",
    help="LMCache remote_url (redis, s3, remote types)",
)
@click.option(
    "--s3-region",
    help="AWS region when --storage-type is s3",
)
@click.option(
    "--extra-config",
    "extra_config_path",
    type=click.Path(exists=True, dir_okay=False),
    help="YAML/JSON merged into generated LMCache config",
)
@click.option(
    "--fs-odirect",
    is_flag=True,
    help="Enable O_DIRECT on LMCache fs:// connector "
    "(filesystem / block-device only); merges "
    "extra_config after --extra-config",
)
@click.option(
    "--probe-remote",
    is_flag=True,
    help="TCP check for redis/lm-style remote_url "
    "before engine start",
)
@click.option(
    "--storage-path",
    help="Storage path (for filesystem or GDS)",
)
@click.option(
    "--block-device",
    help="Block device path (e.g., /dev/nvme0n1)",
)
@click.option(
    "--mount-point",
    help="Mount point for block device",
)
@click.option(
    "--create-fs",
    is_flag=True,
    help="Create filesystem on block device",
)
@click.option(
    "--config",
    default="configs/lmcache-config.yml",
    help="Config file path",
)
@click.option(
    "--chunk-size",
    type=int,
    default=256,
    help="KV cache chunk size",
)
@click.option(
    "--local-cpu",
    is_flag=True,
    help="Enable local CPU caching",
)
@click.option(
    "--max-local-cpu-size",
    type=float,
    default=5.0,
    help="Max local CPU cache size (GB)",
)
@click.option(
    "--cufile-buffer-size",
    type=int,
    default=8192,
    help="CuFile buffer size (MiB) for GDS",
)
@click.option(
    "--model-name",
    default="lmcache_model",
    help="Model name for cache identification",
)
@click.option(
    "--worker-id",
    type=int,
    default=0,
    help="Worker ID",
)
@click.option(
    "--world-size",
    type=int,
    default=1,
    help="Total workers",
)
@click.option(
    "--kv-dtype",
    type=click.Choice(
        ["float16", "float32", "bfloat16", "uint8"]
    ),
    default="float16",
    help="KV cache data type",
)
@click.option(
    "--kv-shape",
    default="2,2,256,4,16",
    help="KV cache shape",
)
@click.option(
    "--use-mla",
    is_flag=True,
    help="Enable Multi-Level Attention",
)
@click.option(
    "--hf-model-name",
    help="Hugging Face model identifier",
)
@click.option(
    "--model-path",
    help="Local path to model",
)
@click.option(
    "--cache-dir",
    help="Directory to cache downloaded models",
)
@click.option(
    "--auto-kv-shape",
    is_flag=True,
    help="Auto-calculate KV shape from model config",
)
@click.option(
    "--local-only",
    is_flag=True,
    help="Only use local models, don't download",
)
@click.option(
    "--hf-token-file",
    help="Path to Hugging Face token file",
)
def verify(
    storage_type: str,
    remote_url: Optional[str],
    s3_region: Optional[str],
    extra_config_path: Optional[str],
    fs_odirect: bool,
    probe_remote: bool,
    storage_path: Optional[str],
    block_device: Optional[str],
    mount_point: Optional[str],
    create_fs: bool,
    config: str,
    chunk_size: int,
    local_cpu: bool,
    max_local_cpu_size: float,
    cufile_buffer_size: int,
    model_name: str,
    worker_id: int,
    world_size: int,
    kv_dtype: str,
    kv_shape: str,
    use_mla: bool,
    hf_model_name: Optional[str],
    model_path: Optional[str],
    cache_dir: Optional[str],
    auto_kv_shape: bool,
    local_only: bool,
    hf_token_file: Optional[str],
):
    """Store then retrieve one chunk; exit 0 on success.

    Success requires retrieve and lookup prefix lengths to match the
    chunk size unless LMVERIFY_RELAXED is set. LMVERIFY_MIN_RETRIEVE
    overrides the minimum retrieve token count (default: chunk size).
    """
    engine_mgr = _engine_manager_class()()
    storage_mgr = StorageManager()
    try:
        config_gen = ConfigGenerator()

        final_kv_shape = kv_shape
        final_kv_dtype = kv_dtype

        if hf_model_name or model_path:
            final_kv_shape, final_kv_dtype = (
                _load_model_params(
                    hf_model_name,
                    model_path,
                    cache_dir,
                    local_only,
                    hf_token_file,
                    chunk_size,
                    kv_dtype,
                    auto_kv_shape,
                )
            )

        config_path = _setup_storage_and_config(
            storage_type,
            storage_path,
            block_device,
            mount_point,
            create_fs,
            config,
            chunk_size,
            local_cpu,
            max_local_cpu_size,
            cufile_buffer_size,
            remote_url,
            s3_region,
            extra_config_path,
            probe_remote,
            fs_odirect,
            storage_mgr,
            config_gen,
        )

        engine_mgr.create_engine(
            config_path=config_path,
            model_name=model_name,
            kv_shape=final_kv_shape,
            kv_dtype=final_kv_dtype,
            worker_id=worker_id,
            world_size=world_size,
            use_mla=use_mla,
        )

        ok = _roundtrip_verify(engine_mgr)
        if ok:
            click.echo("verify: OK")
        else:
            click.echo(
                "verify: FAILED (set LMVERIFY_RELAXED=1 "
                "to allow partial hits)",
                err=True,
            )
        sys.exit(0 if ok else 1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        import traceback

        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)
    finally:
        engine_mgr.close()


@cli.command()
@click.option(
    "--dataset",
    type=click.Choice(
        list(
            _download_conversations
            .DATASET_CONFIGS.keys()
        )
    ),
    default=None,
    help="Dataset to download",
)
@click.option(
    "--output",
    default=None,
    help="Output JSON file path (default: "
         "data/<dataset>-conversations.json)",
)
@click.option(
    "--max-conversations",
    type=int,
    default=500,
    help="Maximum conversations to convert "
         "(default: 500)",
)
@click.option(
    "--reprocess",
    default=None,
    metavar="FILE",
    help="Re-sanitize an existing conversation "
         "JSON file in-place instead of downloading",
)
def download(
    dataset: Optional[str],
    output: Optional[str],
    max_conversations: int,
    reprocess: Optional[str],
):
    """Download and convert chat datasets to the
    LMCache conversation schema."""
    if reprocess:
        _download_conversations.reprocess_file(
            reprocess
        )
        return

    if not dataset:
        click.echo(
            "Error: --dataset is required when "
            "not using --reprocess",
            err=True,
        )
        sys.exit(1)

    out = output or (
        f"data/{dataset}-conversations.json"
    )
    _download_conversations.download_and_convert(
        dataset, out, max_conversations,
    )


def _probe_cache(storage_path: str):
    """Report existing .data files in storage_path
    as a warm-cache indicator."""
    sp = Path(storage_path)
    if not sp.is_dir():
        return
    data_files = list(sp.glob("*.data"))
    if data_files:
        total_bytes = sum(
            f.stat().st_size for f in data_files
        )
        if total_bytes < 1024 ** 2:
            size_str = (
                f"{total_bytes / 1024:.1f} KiB"
            )
        elif total_bytes < 1024 ** 3:
            size_str = (
                f"{total_bytes / 1024**2:.1f} MiB"
            )
        else:
            size_str = (
                f"{total_bytes / 1024**3:.2f} GiB"
            )
        click.echo(
            f"Persist-cache: {len(data_files)} "
            f".data files found ({size_str}) "
            f"in {sp}"
        )
    else:
        click.echo(
            f"Persist-cache: no existing cache "
            f"files in {sp} (cold start)"
        )


def _load_model_params(
    hf_model_name, model_path, cache_dir,
    local_only, hf_token_file, chunk_size,
    kv_dtype, auto_kv_shape,
):
    """Load model and return (kv_shape, kv_dtype)."""
    model_loader = ModelLoader(
        model_name=hf_model_name,
        model_path=model_path,
        cache_dir=cache_dir,
        local_only=local_only,
        token_file=hf_token_file,
    )
    model_loader.load_model()

    kv_params = model_loader.get_kv_cache_params(
        chunk_size=chunk_size, dtype=kv_dtype
    )

    final_kv_dtype = kv_params.dtype
    if auto_kv_shape:
        shape = model_loader.calculate_kv_shape(
            chunk_size=chunk_size, dtype=kv_dtype
        )
        final_kv_shape = ",".join(map(str, shape))
        click.echo(
            f"Auto-calculated KV shape: "
            f"{final_kv_shape}"
        )
    else:
        final_kv_shape = "2,2,256,4,16"

    click.echo(
        f"Loaded model: "
        f"vocab_size={kv_params.vocab_size}, "
        f"num_layers={kv_params.num_layers}, "
        f"num_heads={kv_params.num_heads}"
    )

    return final_kv_shape, final_kv_dtype


def _load_model_params_full(
    hf_model_name, model_path, cache_dir,
    local_only, hf_token_file, chunk_size,
    kv_dtype, auto_kv_shape, tokenizer_mode,
):
    """Load model and return (kv_shape, kv_dtype,
    model_loader)."""
    model_loader = ModelLoader(
        model_name=hf_model_name,
        model_path=model_path,
        cache_dir=cache_dir,
        local_only=local_only,
        token_file=hf_token_file,
    )
    model_loader.load_model()

    kv_params = model_loader.get_kv_cache_params(
        chunk_size=chunk_size, dtype=kv_dtype
    )

    final_kv_dtype = kv_params.dtype
    if auto_kv_shape:
        shape = model_loader.calculate_kv_shape(
            chunk_size=chunk_size, dtype=kv_dtype
        )
        final_kv_shape = ",".join(map(str, shape))
        click.echo(
            f"Auto-calculated KV shape: "
            f"{final_kv_shape}"
        )
    else:
        final_kv_shape = "2,2,256,4,16"

    click.echo(
        f"Loaded model: "
        f"vocab_size={kv_params.vocab_size}, "
        f"num_layers={kv_params.num_layers}, "
        f"num_heads={kv_params.num_heads}"
    )

    return final_kv_shape, final_kv_dtype, model_loader


def _setup_storage_and_config(
    storage_type,
    storage_path,
    block_device,
    mount_point,
    create_fs,
    config,
    chunk_size,
    local_cpu,
    max_local_cpu_size,
    cufile_buffer_size,
    remote_url,
    s3_region,
    extra_config_path,
    probe_remote,
    fs_odirect,
    storage_mgr,
    config_gen,
):
    """Setup storage and generate config file.

    Returns the config file path.
    """
    config_dict = None

    if storage_type == "filesystem":
        if not storage_path:
            click.echo(
                "Error: --storage-path required "
                "for filesystem",
                err=True,
            )
            sys.exit(1)
        is_valid, error = (
            storage_mgr.validate_filesystem_path(
                storage_path
            )
        )
        if not is_valid:
            click.echo(f"Error: {error}", err=True)
            sys.exit(1)

        config_dict = (
            config_gen.generate_filesystem_config(
                storage_path=storage_path,
                chunk_size=chunk_size,
                local_cpu=local_cpu,
                max_local_cpu_size=max_local_cpu_size,
            )
        )

    elif storage_type == "local-disk":
        if not storage_path:
            click.echo(
                "Error: --storage-path required for local-disk "
                "(LMCache local_disk / file://)",
                err=True,
            )
            sys.exit(1)
        Path(storage_path).mkdir(parents=True, exist_ok=True)
        is_valid, error = (
            storage_mgr.validate_filesystem_path(
                storage_path
            )
        )
        if not is_valid:
            click.echo(f"Error: {error}", err=True)
            sys.exit(1)

        config_dict = (
            config_gen.generate_local_disk_config(
                storage_path=storage_path,
                chunk_size=chunk_size,
                local_cpu=local_cpu,
                max_local_cpu_size=max_local_cpu_size,
            )
        )

    elif storage_type == "block-device":
        if not block_device:
            click.echo(
                "Error: --block-device required "
                "for block-device",
                err=True,
            )
            sys.exit(1)

        mount_pt, error = (
            storage_mgr.mount_block_device(
                block_device,
                mount_point,
                create_fs=create_fs,
            )
        )
        if not mount_pt:
            click.echo(f"Error: {error}", err=True)
            sys.exit(1)

        config_dict = (
            config_gen.generate_block_device_config(
                mount_point=mount_pt,
                chunk_size=chunk_size,
                local_cpu=local_cpu,
                max_local_cpu_size=max_local_cpu_size,
            )
        )

    elif storage_type == "gds":
        if not storage_path:
            click.echo(
                "Error: --storage-path required for GDS",
                err=True,
            )
            sys.exit(1)
        is_valid, error = (
            storage_mgr.validate_filesystem_path(
                storage_path
            )
        )
        if not is_valid:
            click.echo(f"Error: {error}", err=True)
            sys.exit(1)

        config_dict = config_gen.generate_gds_config(
            gds_path=storage_path,
            chunk_size=chunk_size,
            local_cpu=local_cpu,
            cufile_buffer_size=cufile_buffer_size,
        )

    elif storage_type == "redis":
        if not remote_url:
            click.echo(
                "Error: --remote-url required for redis "
                "(e.g. redis://127.0.0.1:6379)",
                err=True,
            )
            sys.exit(1)
        ru = remote_url.strip()
        low = ru.lower()
        if not (
            low.startswith("redis://")
            or low.startswith("redis-sentinel://")
        ):
            click.echo(
                "Warning: redis backend expects remote_url "
                "to start with redis:// or "
                "redis-sentinel://",
                err=True,
            )
        if probe_remote:
            ok, msg = storage_mgr.probe_remote_url(ru)
            click.echo(f"probe {msg}")
            if not ok:
                click.echo(
                    "Error: remote probe failed",
                    err=True,
                )
                sys.exit(1)
        config_dict = config_gen.generate_redis_config(
            remote_url=ru,
            chunk_size=chunk_size,
            local_cpu=local_cpu,
            max_local_cpu_size=max_local_cpu_size,
        )

    elif storage_type == "s3":
        if not remote_url or not s3_region:
            click.echo(
                "Error: --remote-url (s3://bucket/...) and "
                "--s3-region are required for s3",
                err=True,
            )
            sys.exit(1)
        ru = remote_url.strip()
        if not ru.lower().startswith("s3://"):
            click.echo(
                "Error: S3 remote_url must start with s3://",
                err=True,
            )
            sys.exit(1)
        config_dict = config_gen.generate_s3_config(
            remote_url=ru,
            s3_region=s3_region.strip(),
            chunk_size=chunk_size,
            local_cpu=local_cpu,
            max_local_cpu_size=max_local_cpu_size,
        )

    elif storage_type == "remote":
        if not remote_url:
            click.echo(
                "Error: --remote-url required for remote "
                "(e.g. lm://127.0.0.1:65432, "
                "mooncakestore://host:port)",
                err=True,
            )
            sys.exit(1)
        ru = remote_url.strip()
        if probe_remote:
            ok, msg = storage_mgr.probe_remote_url(ru)
            click.echo(f"probe {msg}")
            if not ok:
                click.echo(
                    "Error: remote probe failed",
                    err=True,
                )
                sys.exit(1)
        config_dict = config_gen.generate_remote_config(
            remote_url=ru,
            chunk_size=chunk_size,
            local_cpu=local_cpu,
            max_local_cpu_size=max_local_cpu_size,
        )

    else:
        click.echo(
            f"Error: unsupported storage_type "
            f"{storage_type!r}",
            err=True,
        )
        sys.exit(1)

    if extra_config_path:
        config_gen.merge_from_path(
            config_dict,
            extra_config_path,
        )

    if fs_odirect:
        if storage_type in ("filesystem", "block-device"):
            config_gen.apply_fs_odirect_extra_config(
                config_dict
            )
        else:
            click.echo(
                "Warning: --fs-odirect is ignored for "
                f"storage-type={storage_type!r} "
                "(only filesystem and block-device use "
                "the fs:// connector)",
                err=True,
            )

    config_gen.save_config(config_dict, config)
    click.echo(f"Generated config file: {config}")
    return config


def _roundtrip_verify(engine_mgr: Any) -> bool:
    """Store then retrieve one full chunk; check hit counts."""
    shape = engine_mgr._kv_shape
    if shape is None or len(shape) < 3:
        click.echo("Error: engine has no kv_shape", err=True)
        return False
    cs = int(shape[2])
    base = 10_000
    tokens = list(range(base, base + cs))
    req_store = "lmcache-io-verify-store"
    req_ret = "lmcache-io-verify-retrieve"
    engine_mgr.store(tokens, req_id=req_store)
    retrieved = engine_mgr.retrieve(tokens, req_id=req_ret)
    prefix_hits = engine_mgr.lookup(tokens)
    click.echo(
        f"verify: chunk_size={cs} retrieve_hit_tokens="
        f"{retrieved} lookup_prefix_len={prefix_hits}"
    )
    min_ok = int(os.environ.get("LMVERIFY_MIN_RETRIEVE", cs))
    if min_ok < 1:
        min_ok = cs
    relaxed = os.environ.get(
        "LMVERIFY_RELAXED", ""
    ).lower() in ("1", "true", "yes")
    if relaxed:
        ok = retrieved > 0 or prefix_hits > 0
    else:
        ok = retrieved >= min_ok and prefix_hits >= min_ok
    return ok


if __name__ == "__main__":
    cli()
