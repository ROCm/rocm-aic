#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Apply reviewed improvements to grafana/rocm-aic-dashboard.json."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DASH_PATH = REPO / "grafana" / "rocm-aic-dashboard.json"

HIT_RATIO_EXPR = (
    "sum(rate(vllm:external_prefix_cache_hits_total{"
    'instance=~"$vllm_instance",model_name=~"$model"'
    "}[$__rate_interval])) by (instance)\n"
    "/ clamp_min(sum(rate(vllm:external_prefix_cache_queries_total{"
    'instance=~"$vllm_instance",model_name=~"$model"'
    "}[$__rate_interval])) by (instance), 1e-9)"
)

TTFT_P99_EXPR = (
    "histogram_quantile(0.99, sum(rate("
    'vllm:time_to_first_token_seconds_bucket{job="vllm-exporter",'
    'instance=~"$vllm_instance",model_name=~"$model"'
    "}[$__rate_interval])) by (le))"
)

HOST_QUERY = (
    'label_values(up{job=~"node_exporter|node|amd_metrics_exporter|amd_exporter"}, '
    "instance)"
)
VLLM_INSTANCE_QUERY = 'label_values(up{job=~"vllm-exporter|vllm"}, instance)'
RDMA_DEVICE_QUERY = (
    'label_values(rdma_port_rcv_data_total{instance=~"$host"}, device)'
)
BLKDEVICE_QUERY = 'label_values(node_disk_read_bytes_total{instance=~"$host"}, device)'
NFS_MOUNT_DEFAULT = "^$"
NFS_MOUNT_DISABLED_DESC = (
    "Regex for AIC NFS client mount path. Default ^$ disables NFS client "
    "panels (use for local NVMe, hipfile, or other non-NFS AIC). Set to your "
    "mount (for example /mnt/rocm-icms-cache) when the cache is NFS-backed."
)
MODEL_QUERY = "label_values(vllm:e2e_request_latency_seconds_bucket,model_name)"
GPU_ID_QUERY = 'label_values(amd_gpu_power_usage{instance=~"$host"}, gpu_id)'

FREE_PCT_EXPR = (
    '100 * sum(rocm_aic_data_fs_free_bytes{instance=~"$host"})\n'
    '/ clamp_min(sum(rocm_aic_data_fs_total_bytes{instance=~"$host"}), 1)'
)

CHUNK_BYTES_EXPR = 'sum(rocm_aic_kv_chunk_bytes_total{instance=~"$host"})'

NIXL_BYTES_EXPR = 'sum(rocm_aic_nixl_pool_bytes_total{instance=~"$host"})'
NIXL_SLOTS_USED_EXPR = 'sum(rocm_aic_nixl_pool_slots_used{instance=~"$host"})'

CHUNK_HASHES_TRACKED_EXPR = (
    'sum(rocm_aic_chunk_hashes_tracked{instance=~"$host"})'
)
CHUNK_LOOKUP_ROWS_EXPR = (
    'sum(rocm_aic_chunk_stats_lookup_rows{instance=~"$host"})'
)
CHUNK_MENTION_SUM_EXPR = (
    'sum(rocm_aic_chunk_hash_mention_sum{instance=~"$host"})'
)
HOT_CHUNK_PCT_EXPR = (
    "100 * sum(rocm_aic_chunk_hashes_by_lookup_count{instance=~\"$host\","
    'lookup_count=~"11-20|21-50|51-100|>100"})\n'
    '/ clamp_min(sum(rocm_aic_chunk_hashes_tracked{instance=~"$host"}), 1)'
)

ONLINE_GPUS_EXPR = (
    "count by (instance) (\r\n"
    '  amd_gpu_package_power{job="amd_metrics_exporter",instance=~"$host"}\r\n'
    ")"
)

SERVER_POWER_EXPR = '{__name__=~"$server_power_metric"}'

THRESHOLD_GREEN_ONLY = {
    "mode": "absolute",
    "steps": [{"value": 0, "color": "green"}],
}


def prom_query(expr: str, *, instant: bool = False, legend: str = "__auto", fmt: str | None = None) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "editorMode": "code",
        "expr": expr,
        "legendFormat": legend,
        "range": not instant,
    }
    if instant:
        spec["instant"] = True
    if fmt:
        spec["format"] = fmt
    return {
        "kind": "DataQuery",
        "group": "prometheus",
        "version": "v0",
        "datasource": {"name": "${datasource}"},
        "spec": spec,
    }


def panel_query(
    ref_id: str,
    expr: str,
    *,
    instant: bool = False,
    legend: str = "__auto",
    fmt: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": "PanelQuery",
        "spec": {
            "query": prom_query(expr, instant=instant, legend=legend, fmt=fmt),
            "refId": ref_id,
            "hidden": False,
        },
    }


def stat_viz(*, unit: str = "none", min_val: float | None = 0, max_val: float | None = None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "unit": unit,
        "thresholds": THRESHOLD_GREEN_ONLY,
        "color": {"mode": "thresholds"},
    }
    if min_val is not None:
        defaults["min"] = min_val
    if max_val is not None:
        defaults["max"] = max_val
    return {
        "kind": "VizConfig",
        "group": "stat",
        "version": "13.0.1",
        "spec": {
            "options": {
                "colorMode": "value",
                "graphMode": "none",
                "justifyMode": "auto",
                "orientation": "auto",
                "percentChangeColorMode": "standard",
                "reduceOptions": {
                    "calcs": ["lastNotNull"],
                    "fields": "",
                    "values": False,
                },
                "showPercentChange": False,
                "textMode": "auto",
                "wideLayout": True,
            },
            "fieldConfig": {"defaults": defaults, "overrides": []},
        },
    }


def make_stat_panel(
    panel_id: int,
    title: str,
    description: str,
    expr: str,
    *,
    unit: str = "none",
    legend: str = "__auto",
    min_val: float | None = 0,
    max_val: float | None = None,
) -> dict[str, Any]:
    return {
        "kind": "Panel",
        "spec": {
            "id": panel_id,
            "title": title,
            "description": description,
            "links": [],
            "data": {
                "kind": "QueryGroup",
                "spec": {
                    "queries": [panel_query("A", expr, instant=True, legend=legend)],
                    "transformations": [],
                    "queryOptions": {},
                },
            },
            "vizConfig": stat_viz(unit=unit, min_val=min_val, max_val=max_val),
        },
    }


def make_timeseries_panel(
    panel_id: int,
    title: str,
    description: str,
    queries: list[dict[str, Any]],
    *,
    unit: str = "none",
    tooltip_multi: bool = True,
    stacking_mode: str = "none",
    fill_opacity: int = 10,
) -> dict[str, Any]:
    return {
        "kind": "Panel",
        "spec": {
            "id": panel_id,
            "title": title,
            "description": description,
            "links": [],
            "data": {
                "kind": "QueryGroup",
                "spec": {
                    "queries": queries,
                    "transformations": [],
                    "queryOptions": {},
                },
            },
            "vizConfig": {
                "kind": "VizConfig",
                "group": "timeseries",
                "version": "13.0.1",
                "spec": {
                    "options": {
                        "annotations": {"clustering": -1, "multiLane": True},
                        "legend": {
                            "calcs": ["min", "mean", "max", "lastNotNull"],
                            "displayMode": "table",
                            "placement": "bottom",
                            "showLegend": True,
                        },
                        "tooltip": {
                            "hideZeros": False,
                            "mode": "multi" if tooltip_multi else "single",
                            "sort": "desc",
                        },
                    },
                    "fieldConfig": {
                        "defaults": {
                            "unit": unit,
                            "min": 0,
                            "thresholds": THRESHOLD_GREEN_ONLY,
                            "color": {"mode": "palette-classic"},
                            "custom": {
                                "axisBorderShow": False,
                                "axisCenteredZero": False,
                                "axisColorMode": "text",
                                "axisLabel": "",
                                "axisPlacement": "auto",
                                "barAlignment": 0,
                                "barWidthFactor": 0.6,
                                "drawStyle": "line",
                                "fillOpacity": fill_opacity,
                                "gradientMode": "none",
                                "hideFrom": {
                                    "legend": False,
                                    "tooltip": False,
                                    "viz": False,
                                },
                                "insertNulls": False,
                                "lineInterpolation": "linear",
                                "lineWidth": 1,
                                "pointSize": 0,
                                "scaleDistribution": {"type": "linear"},
                                "showPoints": "never",
                                "showValues": False,
                                "spanNulls": False,
                                "stacking": {"group": "A", "mode": stacking_mode},
                                "thresholdsStyle": {"mode": "off"},
                            },
                        },
                        "overrides": [],
                    },
                },
            },
        },
    }


def grid_item(name: str, x: int, y: int, width: int, height: int) -> dict[str, Any]:
    return {
        "kind": "GridLayoutItem",
        "spec": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "element": {"kind": "ElementReference", "name": name},
        },
    }


def optional_constant_var(
    name: str,
    label: str,
    description: str,
    *,
    hide: str = "dontHide",
) -> dict[str, Any]:
    return {
        "kind": "ConstantVariable",
        "spec": {
            "name": name,
            "current": {"text": NFS_MOUNT_DEFAULT, "value": NFS_MOUNT_DEFAULT},
            "label": label,
            "hide": hide,
            "skipUrlSync": False,
            "description": description,
            "query": NFS_MOUNT_DEFAULT,
            "options": [
                {"text": NFS_MOUNT_DEFAULT, "value": NFS_MOUNT_DEFAULT, "selected": True}
            ],
            "multi": False,
            "includeAll": False,
            "allowCustomValue": True,
        },
    }


def query_var(
    name: str,
    label: str,
    query: str,
    description: str = "",
    multi: bool = False,
    include_all: bool = False,
    sort: str = "alphabeticalAsc",
    hide: str = "dontHide",
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "name": name,
        "current": {"text": "All", "value": "$__all"} if include_all else {"text": "", "value": ""},
        "label": label,
        "hide": hide,
        "refresh": "onTimeRangeChanged",
        "skipUrlSync": False,
        "description": description,
        "query": {
            "kind": "DataQuery",
            "group": "prometheus",
            "version": "v0",
            "datasource": {"name": "${datasource}"},
            "spec": {
                "qryType": 1,
                "query": query,
                "refId": "PrometheusVariableQueryEditor-VariableQuery",
            },
        },
        "regex": "",
        "regexApplyTo": "value",
        "sort": sort,
        "definition": query,
        "options": [],
        "multi": multi,
        "includeAll": include_all,
        "allowCustomValue": False,
    }
    if include_all:
        spec["allValue"] = ".+"
    return {"kind": "QueryVariable", "spec": spec}


def panel_by_id(elements: dict[str, Any], panel_id: int) -> dict[str, Any] | None:
    for elem in elements.values():
        if elem.get("kind") == "Panel" and elem.get("spec", {}).get("id") == panel_id:
            return elem
    return None


def set_expr(panel: dict[str, Any], ref_id: str | None, expr: str) -> None:
    queries = panel["spec"]["data"]["spec"]["queries"]
    for q in queries:
        if ref_id is None or q["spec"]["refId"] == ref_id:
            q["spec"]["query"]["spec"]["expr"] = expr


def set_query_instant(panel: dict[str, Any], instant: bool = True) -> None:
    for q in panel["spec"]["data"]["spec"]["queries"]:
        qspec = q["spec"]["query"]["spec"]
        qspec["instant"] = instant
        qspec["range"] = not instant


def set_unit(panel: dict[str, Any], unit: str, *, max_val: float | None = None) -> None:
    defaults = panel["spec"]["vizConfig"]["spec"]["fieldConfig"]["defaults"]
    defaults["unit"] = unit
    defaults["thresholds"] = copy.deepcopy(THRESHOLD_GREEN_ONLY)
    if max_val is not None:
        defaults["max"] = max_val
    elif "max" in defaults and defaults.get("unit") not in ("percent", "percentunit"):
        defaults.pop("max", None)


def set_tooltip_multi(panel: dict[str, Any]) -> None:
    panel["spec"]["vizConfig"]["spec"]["options"]["tooltip"]["mode"] = "multi"
    panel["spec"]["vizConfig"]["spec"]["options"]["tooltip"]["sort"] = "desc"


def clear_bad_thresholds(panel: dict[str, Any], *, keep_percent: bool = False) -> None:
    defaults = panel["spec"]["vizConfig"]["spec"]["fieldConfig"]["defaults"]
    unit = defaults.get("unit", "")
    if keep_percent and unit in ("percent", "percentunit"):
        steps = defaults.get("thresholds", {}).get("steps", [])
        if len(steps) >= 2 and steps[1].get("value") == 80:
            return
    defaults["thresholds"] = copy.deepcopy(THRESHOLD_GREEN_ONLY)


def fix_variables(variables: list[dict[str, Any]]) -> None:
    """Patch legacy variable entries (rebuild_variables replaces the full list)."""
    all_current = {"text": "All", "value": "$__all"}
    for var in variables:
        spec = var.get("spec", {})
        name = spec.get("name")
        if name == "datasource":
            spec["current"] = {}
            spec["options"] = []
            spec["allowCustomValue"] = True
        if name in (
            "host",
            "vllm_instance",
            "model",
            "gpu_id",
            "blkdevice",
            "rdma_device",
            "nfs_mount",
        ):
            spec["current"] = copy.deepcopy(all_current)
            spec["refresh"] = "onTimeRangeChanged"
        if name == "server_power_metric":
            spec["hide"] = "hideVariable"
            spec["current"] = {"text": "^$", "value": "^$"}
            spec["query"] = "^$"
            spec["options"] = [{"text": "^$", "value": "^$", "selected": True}]


def rebuild_variables() -> list[dict[str, Any]]:
    return [
        {
            "kind": "DatasourceVariable",
            "spec": {
                "name": "datasource",
                "pluginId": "prometheus",
                "refresh": "onDashboardLoad",
                "regex": "",
                "current": {},
                "options": [],
                "multi": False,
                "includeAll": False,
                "label": "Data source",
                "hide": "hideVariable",
                "skipUrlSync": False,
                "allowCustomValue": True,
            },
        },
        query_var(
            "host",
            "Host",
            HOST_QUERY,
            "Prometheus instance for node_exporter and amd_metrics_exporter "
            "(Ansible clusters use the inventory hostname for both).",
            multi=True,
            include_all=True,
        ),
        query_var(
            "vllm_instance",
            "vLLM Instance",
            VLLM_INSTANCE_QUERY,
            "vLLM exporter target (host:port) for latency and token metrics.",
            multi=True,
            include_all=True,
        ),
        query_var(
            "model",
            "LLM Model",
            MODEL_QUERY,
            "LLM model_name label on vLLM metrics.",
            multi=True,
            include_all=True,
            sort="disabled",
        ),
        query_var(
            "gpu_id",
            "GPU ID",
            GPU_ID_QUERY,
            "GPU id from amd_metrics_exporter on the selected host(s).",
            multi=True,
            include_all=True,
            sort="disabled",
        ),
        query_var(
            "blkdevice",
            "Block Device",
            BLKDEVICE_QUERY,
            "Block device on the selected node(s) for KV storage bandwidth.",
            multi=True,
            include_all=True,
        ),
        query_var(
            "rdma_device",
            "RDMA Device",
            RDMA_DEVICE_QUERY,
            "RDMA port on the selected node(s); used when the storage backend "
            "is reachable over RoCE (optional).",
            multi=True,
            include_all=True,
        ),
        optional_constant_var(
            "nfs_mount",
            "NFS Mount (optional)",
            NFS_MOUNT_DISABLED_DESC,
        ),
        optional_constant_var(
            "server_power_metric",
            "Server Power Metric",
            "Regex for optional PDU power metric; ^$ disables the series.",
            hide="hideVariable",
        ),
    ]


def normalize_host_variable_refs(dash: dict[str, Any]) -> None:
    """Map legacy $node / $instance panel variables to unified $host."""

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if "expr" in obj and isinstance(obj["expr"], str):
                expr = obj["expr"]
                expr = expr.replace("$node", "$host")
                expr = expr.replace('instance=~"$instance"', 'instance=~"$host"')
                obj["expr"] = expr
            else:
                for v in obj.values():
                    walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(dash)


def fix_host_metric_exprs(dash: dict[str, Any]) -> None:
    replacements = [
        ('mount_point="$nfs_mount"', 'mount_point=~"$nfs_mount"'),
    ]
    host_only_prefixes = ("rocm_aic_", "node_disk_", "node_nfsd_")

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if "expr" in obj and isinstance(obj["expr"], str):
                expr = obj["expr"]
                if any(p in expr for p in host_only_prefixes):
                    for old, new in replacements:
                        expr = expr.replace(old, new)
                    obj["expr"] = expr
            else:
                for v in obj.values():
                    walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(dash)


def apply_improvements(dash: dict[str, Any]) -> None:
    spec = dash["spec"]
    elements = spec["elements"]

    # --- Query and panel fixes on existing panels ---
    p31 = panel_by_id(elements, 31)
    if p31:
        p31["spec"]["description"] = (
            "GPU power from amd_metrics_exporter. Server PDU power appears only "
            "when server_power_metric is set (Dashboard settings → Variables)."
        )
        set_expr(p31, "server_power_metric-avg", SERVER_POWER_EXPR)
        set_expr(p31, "A", 'amd_gpu_power_usage{instance=~"$host",gpu_id=~"$gpu_id"}')
        for q in p31["spec"]["data"]["spec"]["queries"]:
            if q["spec"]["refId"] == "A":
                q["spec"]["query"]["spec"]["legendFormat"] = "{{instance}} GPU {{gpu_id}}"
        clear_bad_thresholds(p31)

    p32 = panel_by_id(elements, 32)
    if p32:
        for q in p32["spec"]["data"]["spec"]["queries"]:
            q["spec"]["query"]["spec"]["legendFormat"] = "{{instance}} GPU {{gpu_id}}"
        set_unit(p32, "percent", max_val=100)
        set_tooltip_multi(p32)

    p34 = panel_by_id(elements, 34)
    if p34:
        for q in p34["spec"]["data"]["spec"]["queries"]:
            q["spec"]["query"]["spec"]["legendFormat"] = "{{instance}}"
        clear_bad_thresholds(p34)

    p35 = panel_by_id(elements, 35)
    if p35:
        p35["spec"]["title"] = "KV Cache Block I/O (local device)"
        p35["spec"]["description"] = (
            "Read/write bandwidth on the selected block device(s). Use for "
            "local NVMe, dm-crypt, or other direct-attached AIC storage — "
            "not NFS client traffic (see optional NFS panels below)."
        )
        set_expr(
            p35,
            "node_disk_written_bytes_total-sum(rate)",
            'sum(rate(node_disk_read_bytes_total{instance=~"$host",device=~"$blkdevice"}[$__rate_interval]))',
        )
        set_expr(
            p35,
            "A",
            'sum(rate(node_disk_written_bytes_total{instance=~"$host",device=~"$blkdevice"}[$__rate_interval]))',
        )
        set_unit(p35, "binBps")
        set_tooltip_multi(p35)

    p44 = panel_by_id(elements, 44)
    if p44:
        set_expr(p44, "A", 'rocm_aic_kv_disk_files{instance=~"$host"}')

    p46 = panel_by_id(elements, 46)
    if p46:
        set_expr(p46, "A", 'rocm_aic_data_fs_free_bytes{instance=~"$host"}')
        set_unit(p46, "bytes")
        clear_bad_thresholds(p46)

    p48 = panel_by_id(elements, 48)
    if p48:
        p48["spec"]["description"] = (
            "RDMA throughput to the storage backend when RoCE is in use "
            "(optional; not applicable to all AIC deployments)."
        )
        set_unit(p48, "binBps")
        set_tooltip_multi(p48)

    p51 = panel_by_id(elements, 51)
    if p51:
        p51["spec"]["title"] = "NFS Server Bandwidth (optional)"
        p51["spec"]["description"] = (
            "Kernel NFS server read/write throughput on the selected node. "
            "Only populated on nodes that export the AIC cache; expect No "
            "data on GPU clients or when AIC is local/NVMe."
        )
        set_expr(
            p51,
            "A",
            'rate(node_nfsd_disk_bytes_read_total{instance=~"$host"}[$__rate_interval])',
        )
        set_expr(
            p51,
            "B",
            'rate(node_nfsd_disk_bytes_written_total{instance=~"$host"}[$__rate_interval])',
        )
        set_unit(p51, "binBps")

    p52 = panel_by_id(elements, 52)
    if p52:
        p52["spec"]["title"] = "NFS Client Bandwidth (optional)"
        p52["spec"]["description"] = (
            "NFS client throughput for the AIC cache mount (rocm_aic_nfs_mount_*). "
            "Set nfs_mount to your mount path when AIC is NFS-backed; leave nfs_mount "
            "as ^$ for local NVMe, hipfile, or other non-NFS storage."
        )
        set_expr(
            p52,
            "A",
            'rate(rocm_aic_nfs_mount_rx_bytes_total{instance=~"$host",mount_point=~"$nfs_mount"}[$__rate_interval])',
        )
        set_expr(
            p52,
            "B",
            'rate(rocm_aic_nfs_mount_tx_bytes_total{instance=~"$host",mount_point=~"$nfs_mount"}[$__rate_interval])',
        )
        set_unit(p52, "binBps")

    p47 = panel_by_id(elements, 47)
    if p47:
        p47["spec"]["title"] = "KV Cache Block Hit Statistics (.data)"
        p47["spec"]["description"] = (
            "Hit distribution across on-disk LMCache .data files. NIXL-only "
            "deployments (obj_*.bin, no .data) use the NIXL Chunk Lookup "
            "Stats row below instead."
        )
        set_expr(
            p47,
            "A",
            'sum(\r\n  rate(rocm_aic_kv_files_by_hit_count{instance=~"$host",hit_count=~"0|1"}[$__rate_interval])\r\n) by (instance)',
        )
        set_expr(
            p47,
            "C",
            'sum(\r\n  rate(rocm_aic_kv_files_by_hit_count{instance=~"$host",hit_count=~"2|3|4|5"}[$__rate_interval])\r\n) by (instance)',
        )

    p43 = panel_by_id(elements, 43)
    if p43:
        set_expr(
            p43,
            "A",
            'sum(rate(vllm:prompt_tokens_by_source_total{instance=~"$vllm_instance",'
            'model_name=~"$model",source="local_compute"}[$__rate_interval]))',
        )
        set_tooltip_multi(p43)

    p5 = panel_by_id(elements, 5)
    if p5:
        set_expr(p5, "A", HIT_RATIO_EXPR)
        set_unit(p5, "percentunit", max_val=1)
        set_tooltip_multi(p5)

    p6 = panel_by_id(elements, 6)
    if p6:
        p6["spec"]["description"] = (
            "Time-to-first-token from vllm:time_to_first_token_seconds_bucket; "
            "p50, p90, and p99 over $__rate_interval."
        )
        set_tooltip_multi(p6)

    p49 = panel_by_id(elements, 49)
    if p49:
        set_tooltip_multi(p49)

    p50 = panel_by_id(elements, 50)
    if p50:
        set_tooltip_multi(p50)

    p37 = panel_by_id(elements, 37)
    if p37:
        set_expr(p37, "A", "lmcache:num_store_requests_total")
        set_expr(p37, "B", "lmcache:num_retrieve_requests_total")

    p53 = panel_by_id(elements, 53)
    if p53:
        set_query_instant(p53, True)
        clear_bad_thresholds(p53)
        set_expr(p53, "A", 'rocm_aic_rocm_version_info{instance=~"$host"}')
        p53["spec"]["data"]["spec"]["transformations"] = []
        for q in p53["spec"]["data"]["spec"]["queries"]:
            q["spec"]["query"]["spec"]["format"] = "table"

    p54 = panel_by_id(elements, 54)
    if p54:
        set_expr(p54, "A", ONLINE_GPUS_EXPR)
        p54["spec"]["description"] = (
            "GPUs reporting amd_gpu_package_power on the selected node(s)."
        )
        set_unit(p54, "none")
        p54["spec"]["vizConfig"]["spec"]["options"]["graphMode"] = "none"
        clear_bad_thresholds(p54)

    # Strip bogus 80 thresholds from remaining timeseries panels.
    for elem in elements.values():
        if elem.get("kind") != "Panel":
            continue
        pid = elem["spec"].get("id")
        if pid in (5, 32):
            continue
        viz = elem["spec"].get("vizConfig", {})
        if viz.get("group") == "timeseries":
            clear_bad_thresholds(elem)
        elif viz.get("group") == "table":
            clear_bad_thresholds(elem)

    # --- New overview + storage panels ---
    elements["panel-55"] = make_stat_panel(
        55,
        "TTFT p99",
        "Latest p99 time-to-first-token for the selected vLLM target(s) and model(s).",
        TTFT_P99_EXPR,
        unit="s",
    )
    elements["panel-56"] = make_stat_panel(
        56,
        "AIC Prefix Hit Ratio",
        "External prefix cache hit ratio (LMCache / KV connector) for selected vLLM instances.",
        HIT_RATIO_EXPR,
        unit="percentunit",
        max_val=1,
    )
    elements["panel-57"] = make_stat_panel(
        57,
        "KV Chunk Bytes",
        "Total on-disk LMCache .data chunk bytes (rocm_aic_kv_chunk_bytes_total).",
        CHUNK_BYTES_EXPR,
        unit="bytes",
    )
    elements["panel-58"] = make_stat_panel(
        58,
        "AIC Storage Free %",
        "Free space percentage on the AIC data filesystem.",
        FREE_PCT_EXPR,
        unit="percent",
        max_val=100,
    )
    elements["panel-59"] = make_timeseries_panel(
        59,
        "AIC KV Cache Footprint",
        "On-disk LMCache chunk and NIXL static pool size over time.",
        [
            panel_query("A", CHUNK_BYTES_EXPR, legend="KV chunk bytes"),
            panel_query("B", NIXL_BYTES_EXPR, legend="NIXL pool bytes"),
        ],
        unit="bytes",
    )

    elements["panel-60"] = make_stat_panel(
        60,
        "Distinct Chunk Hashes",
        "Unique chunk hashes seen in chunk_hashes JSONL (NIXL and .data modes).",
        CHUNK_HASHES_TRACKED_EXPR,
        unit="none",
    )
    elements["panel-61"] = make_stat_panel(
        61,
        "Chunk Lookup Rows",
        "chunk_hashes JSONL rows scanned by rocm-aic-exporter.",
        CHUNK_LOOKUP_ROWS_EXPR,
        unit="none",
    )
    elements["panel-62"] = make_stat_panel(
        62,
        "Hash Mention Sum",
        "Total chunk hash mentions across all JSONL lookup rows.",
        CHUNK_MENTION_SUM_EXPR,
        unit="none",
    )
    elements["panel-63"] = make_stat_panel(
        63,
        "Hot Chunks (>10 lookups)",
        "Percentage of distinct chunk hashes with more than 10 JSONL lookup "
        "mentions (sum of 11-20, 21-50, 51-100, and >100 buckets).",
        HOT_CHUNK_PCT_EXPR,
        unit="percent",
        max_val=100,
    )
    elements["panel-64"] = make_timeseries_panel(
        64,
        "NIXL Chunk Lookup Distribution",
        "How many distinct chunk hashes fall into each JSONL lookup mention "
        "bucket. Populated for NIXL (obj_*.bin) and hipfile (.data) when "
        "chunk statistics are enabled; measures lookup references per hash, "
        "not per NIXL pool slot.",
        [
            panel_query(
                "A",
                'sum(rocm_aic_chunk_hashes_by_lookup_count{instance=~"$host",'
                'lookup_count=~"1|2"}) by (instance)',
                legend="1-2 mentions",
            ),
            panel_query(
                "B",
                'sum(rocm_aic_chunk_hashes_by_lookup_count{instance=~"$host",'
                'lookup_count=~"3|4|5"}) by (instance)',
                legend="3-5 mentions",
            ),
            panel_query(
                "C",
                'sum(rocm_aic_chunk_hashes_by_lookup_count{instance=~"$host",'
                'lookup_count=~"6|7|8|9|10"}) by (instance)',
                legend="6-10 mentions",
            ),
            panel_query(
                "D",
                'sum(rocm_aic_chunk_hashes_by_lookup_count{instance=~"$host",'
                'lookup_count="11-20"}) by (instance)',
                legend="11-20 mentions",
            ),
            panel_query(
                "E",
                'sum(rocm_aic_chunk_hashes_by_lookup_count{instance=~"$host",'
                'lookup_count="21-50"}) by (instance)',
                legend="21-50 mentions",
            ),
            panel_query(
                "F",
                'sum(rocm_aic_chunk_hashes_by_lookup_count{instance=~"$host",'
                'lookup_count="51-100"}) by (instance)',
                legend="51-100 mentions",
            ),
            panel_query(
                "G",
                'sum(rocm_aic_chunk_hashes_by_lookup_count{instance=~"$host",'
                'lookup_count=">100"}) by (instance)',
                legend=">100 mentions",
            ),
        ],
        unit="none",
        stacking_mode="normal",
        fill_opacity=25,
    )

    # --- Layout: overview stats, then grouped sections ---
    spec["layout"] = {
        "kind": "GridLayout",
        "spec": {
            "items": [
                grid_item("panel-54", 0, 0, 4, 4),
                grid_item("panel-55", 4, 0, 5, 4),
                grid_item("panel-56", 9, 0, 5, 4),
                grid_item("panel-57", 14, 0, 5, 4),
                grid_item("panel-58", 19, 0, 5, 4),
                grid_item("panel-53", 0, 4, 24, 5),
                grid_item("panel-32", 0, 9, 8, 12),
                grid_item("panel-34", 8, 9, 8, 12),
                grid_item("panel-31", 16, 9, 8, 12),
                grid_item("panel-5", 0, 21, 12, 10),
                grid_item("panel-43", 12, 21, 12, 10),
                grid_item("panel-6", 0, 31, 8, 10),
                grid_item("panel-49", 8, 31, 8, 10),
                grid_item("panel-50", 16, 31, 8, 10),
                grid_item("panel-35", 0, 41, 8, 10),
                grid_item("panel-48", 8, 41, 8, 10),
                grid_item("panel-47", 16, 41, 8, 10),
                grid_item("panel-44", 0, 51, 8, 8),
                grid_item("panel-59", 8, 51, 8, 8),
                grid_item("panel-46", 16, 51, 8, 8),
                grid_item("panel-52", 0, 59, 12, 8),
                grid_item("panel-51", 12, 59, 12, 8),
                grid_item("panel-37", 0, 67, 24, 8),
                grid_item("panel-60", 0, 75, 6, 4),
                grid_item("panel-61", 6, 75, 6, 4),
                grid_item("panel-62", 12, 75, 6, 4),
                grid_item("panel-63", 18, 75, 6, 4),
                grid_item("panel-64", 0, 79, 24, 10),
            ]
        },
    }

    spec["description"] = (
        "ROCm AMD Infinity Context System: KV cache storage, LMCache, vLLM "
        "inference, and GPU metrics. AIC storage may be local block device, "
        "hipfile, NFS, or other backends — use optional NFS panels only when "
        "the cache is NFS-mounted."
    )
    spec["timeSettings"]["from"] = "now-6h"
    spec["timeSettings"]["autoRefresh"] = "30s"

    normalize_host_variable_refs(dash)
    fix_host_metric_exprs(dash)
    spec["variables"] = rebuild_variables()


def main() -> None:
    dash = json.loads(DASH_PATH.read_text(encoding="utf-8"))
    apply_improvements(dash)
    DASH_PATH.write_text(json.dumps(dash, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated {DASH_PATH}")


if __name__ == "__main__":
    main()
