#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Merge working rocm-aic-dashboard.json with useful committed panels."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from dashboard_v2_wire import fix_dashboard_v2_wire_format  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DASH_PATH = REPO / "grafana" / "rocm-aic-dashboard.json"
COMMITTED_PATH = REPO / ".git" / "HEAD"

THRESHOLD_GREEN = {
    "mode": "absolute",
    "steps": [{"value": 0, "color": "green"}],
}

GPU_ACTIVITY_BY_ID_EXPR = (
    "avg by (gpu_id) (\r\n"
    "  amd_gpu_gfx_activity{server_name=\"$instance\"}\r\n"
    "  + on(gpu_id, server_name) group_left\r\n"
    "  amd_gpu_umc_activity{server_name=\"$instance\"}\r\n"
    ")"
)

ONLINE_GPUS_EXPR = (
    'count(amd_gpu_gfx_activity{server_name="$instance"})'
)

TTFT_P99_EXPR = (
    "histogram_quantile(0.99, sum(rate("
    'vllm:time_to_first_token_seconds_bucket{instance=~"$instance"}'
    "[$__rate_interval])) by (le))"
)

HIT_RATIO_EXPR = (
    "sum(rate(vllm:external_prefix_cache_hits_total{"
    'instance=~"$instance"}[$__rate_interval])) by (instance)\n'
    "/ clamp_min(sum(rate(vllm:external_prefix_cache_queries_total{"
    'instance=~"$instance"}[$__rate_interval])) by (instance), 1e-9)'
)

FREE_PCT_EXPR = (
    '100 * sum(rocm_aic_data_fs_free_bytes{server_name="$instance"})\n'
    '/ clamp_min(sum(rocm_aic_data_fs_total_bytes{server_name="$instance"}), 1)'
)

CHUNK_BYTES_EXPR = (
    'sum(rocm_aic_kv_chunk_bytes_total{server_name="$instance"})'
)


def prom_query(
    expr: str,
    *,
    instant: bool = False,
    legend: str = "__auto",
    fmt: str | None = None,
) -> dict[str, Any]:
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
        "thresholds": copy.deepcopy(THRESHOLD_GREEN),
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


def bargauge_viz(*, unit: str = "percent", max_val: float = 100) -> dict[str, Any]:
    return {
        "kind": "VizConfig",
        "group": "bargauge",
        "version": "13.0.1",
        "spec": {
            "options": {
                "displayMode": "gradient",
                "legend": {
                    "calcs": [],
                    "displayMode": "list",
                    "placement": "bottom",
                    "showLegend": False,
                },
                "minVizHeight": 16,
                "minVizWidth": 8,
                "namePlacement": "auto",
                "orientation": "horizontal",
                "reduceOptions": {
                    "calcs": ["mean"],
                    "fields": "",
                    "values": False,
                },
                "showUnfilled": True,
                "sizing": "auto",
                "valueMode": "color",
            },
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "min": 0,
                    "max": max_val,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "blue"},
                            {"value": 25, "color": "green"},
                            {"value": 50, "color": "yellow"},
                            {"value": 75, "color": "orange"},
                            {"value": 90, "color": "red"},
                        ],
                    },
                    "color": {"mode": "thresholds"},
                },
                "overrides": [],
            },
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


def make_bargauge_panel(
    panel_id: int,
    title: str,
    description: str,
    expr: str,
    *,
    legend: str = "__auto",
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
                    "queries": [panel_query("A", expr, legend=legend)],
                    "transformations": [
                        {
                            "kind": "sortBy",
                            "spec": {
                                "id": "sortBy",
                                "options": {
                                    "fields": {},
                                    "sort": [{"desc": True, "field": "Value"}],
                                },
                            },
                        }
                    ],
                    "queryOptions": {"maxDataPoints": 500},
                },
            },
            "vizConfig": bargauge_viz(),
        },
    }


def make_table_panel(
    panel_id: int,
    title: str,
    description: str,
    expr: str,
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
                    "queries": [
                        panel_query("A", expr, instant=True, fmt="table"),
                    ],
                    "transformations": [],
                    "queryOptions": {},
                },
            },
            "vizConfig": {
                "kind": "VizConfig",
                "group": "table",
                "version": "13.0.1",
                "spec": {
                    "options": {"cellHeight": "sm", "showHeader": True},
                    "fieldConfig": {
                        "defaults": {
                            "thresholds": copy.deepcopy(THRESHOLD_GREEN),
                            "color": {"mode": "thresholds"},
                            "custom": {
                                "align": "auto",
                                "cellOptions": {"type": "auto"},
                                "footer": {"reducers": []},
                                "inspect": False,
                            },
                        },
                        "overrides": [],
                    },
                },
            },
        },
    }


def fix_datasource_refs(obj: Any) -> None:
    if isinstance(obj, dict):
        for key, val in list(obj.items()):
            if key == "name" and val == "ae8pyuqoyonpca":
                obj[key] = "${datasource}"
            elif isinstance(val, str) and "ae8pyuqoyonpca" in val:
                obj[key] = val.replace("ae8pyuqoyonpca", "${datasource}")
            else:
                fix_datasource_refs(val)
    elif isinstance(obj, list):
        for item in obj:
            fix_datasource_refs(item)


def clear_red_threshold(panel: dict[str, Any]) -> None:
    defaults = panel["spec"]["vizConfig"]["spec"]["fieldConfig"]["defaults"]
    steps = defaults.get("thresholds", {}).get("steps", [])
    if len(steps) >= 2 and steps[1].get("value") == 80:
        defaults["thresholds"] = copy.deepcopy(THRESHOLD_GREEN)


def layout_ref(name: str) -> dict[str, Any]:
    return {
        "kind": "AutoGridLayoutItem",
        "spec": {
            "element": {"kind": "ElementReference", "name": name},
        },
    }


def merge_dashboard(working: dict[str, Any]) -> dict[str, Any]:
    dash = copy.deepcopy(working)
    spec = dash["spec"]
    elements = spec["elements"]

    fix_datasource_refs(dash)

    # Fix incomplete / untitled panels from the working export.
    if "panel-53" in elements:
        p53 = elements["panel-53"]["spec"]
        p53["title"] = "KV Cache Utilization"
        p53["description"] = (
            "Share of KV-cache blocks in use on the selected instance "
            "(vllm:kv_cache_usage_perc)."
        )

    if "panel-55" in elements:
        p55 = elements["panel-55"]["spec"]
        p55["title"] = "AIC Storage Usage Rate"
        p55["description"] = (
            "Rate of change in AIC data filesystem used bytes by mount path "
            "on the selected instance."
        )

    for key in ("panel-31", "panel-32", "panel-34"):
        elem = elements.get(key)
        if elem:
            clear_red_threshold(elem)

    # GPU activity ranking — helps pick gpu_id from the variable menu.
    elements["panel-56"] = make_bargauge_panel(
        56,
        "GPU Activity by ID",
        "Mean GFX + UMC activity per GPU over the dashboard time range on "
        "the selected instance. Use the busiest gpu_id in the GPU ID "
        "variable when vLLM spans multiple GPUs.",
        GPU_ACTIVITY_BY_ID_EXPR,
        legend="GPU {{gpu_id}}",
    )

    # Overview stats from the committed dashboard, adapted to server_name.
    elements["panel-57"] = make_stat_panel(
        57,
        "Online GPUs",
        "GPUs reporting amd_gpu_gfx_activity on the selected instance.",
        ONLINE_GPUS_EXPR,
        unit="none",
        min_val=None,
    )
    elements["panel-58"] = make_stat_panel(
        58,
        "TTFT p99",
        "Latest p99 time-to-first-token for the selected vLLM instance.",
        TTFT_P99_EXPR,
        unit="s",
    )
    elements["panel-59"] = make_stat_panel(
        59,
        "AIC Prefix Hit Ratio",
        "External prefix cache hit ratio (LMCache / KV connector).",
        HIT_RATIO_EXPR,
        unit="percentunit",
        max_val=1,
    )
    elements["panel-60"] = make_stat_panel(
        60,
        "AIC Storage Free %",
        "Free space percentage on the AIC data filesystem.",
        FREE_PCT_EXPR,
        unit="percent",
        max_val=100,
    )
    elements["panel-61"] = make_stat_panel(
        61,
        "KV Chunk Bytes",
        "Total on-disk LMCache .data chunk bytes "
        "(rocm_aic_kv_chunk_bytes_total).",
        CHUNK_BYTES_EXPR,
        unit="bytes",
        min_val=None,
    )
    elements["panel-62"] = make_table_panel(
        62,
        "ROCm Version",
        "ROCm version reported by rocm-aic-exporter on the selected instance.",
        'rocm_aic_rocm_version_info{server_name="$instance"}',
    )

    overview_panels = [
        "panel-56",
        "panel-57",
        "panel-58",
        "panel-59",
        "panel-60",
        "panel-61",
        "panel-62",
    ]
    existing_items = spec["layout"]["spec"]["items"]
    spec["layout"]["spec"]["items"] = (
        [layout_ref(name) for name in overview_panels] + existing_items
    )

    spec["description"] = (
        "ROCm AMD Infinity Context System: KV cache storage, LMCache, vLLM "
        "inference, GPU metrics, and AIS I/O. AIC storage may be local block "
        "device, hipfile, NFS, or other backends. Use GPU Activity by ID to "
        "pick the busiest gpu_id on multi-GPU nodes."
    )
    spec["timeSettings"]["from"] = "now-6h"
    spec["timeSettings"]["autoRefresh"] = "30s"

    dash["metadata"] = {
        "name": "rocm-aic-dashboard",
        "namespace": "default",
        "uid": "7f2af756-9968-44cd-8f7a-cd8be435dd28",
    }

    fix_dashboard_v2_wire_format(dash)
    return dash


def main() -> None:
    working = json.loads(DASH_PATH.read_text(encoding="utf-8"))
    merged = merge_dashboard(working)
    DASH_PATH.write_text(
        json.dumps(merged, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Merged dashboard written to {DASH_PATH}")


if __name__ == "__main__":
    main()
