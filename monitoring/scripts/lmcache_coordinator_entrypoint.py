#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Combined entrypoint for the LMCache MP coordinator + Prometheus exporter.
#
# Runs the coordinator (lmcache.v1.mp_coordinator) in a background thread via
# uvicorn, and serves /metrics (Prometheus text format) on a second port in the
# same process.  A single python:3.12-slim container replaces the two heavy
# rocm-aic containers that were previously used.
#
# Env vars (docker/docker-compose.yml sets every one and owns the defaults):
#   LMCACHE_COORDINATOR_PORT        coordinator HTTP port
#   COORDINATOR_METRICS_PORT        /metrics port
#   LMCACHE_COORDINATOR_INSTANCE_TIMEOUT   seconds before stale instance eviction
#   LMCACHE_COORDINATOR_HEALTH_INTERVAL    health-check sweep interval
#   LMCACHE_COORDINATOR_EVICTION_INTERVAL  L2 eviction sweep interval
#   LMCACHE_COORDINATOR_KEEPALIVE          uvicorn keep-alive timeout
#   LMCACHE_CHUNK_SIZE                     tokens per KV chunk
#   LMCACHE_DISABLE_BANNER                 set to 1 to suppress the lmcache banner

import http.server
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("coordinator")

COORD_PORT = int(os.environ["LMCACHE_COORDINATOR_PORT"])
METRICS_PORT = int(os.environ["COORDINATOR_METRICS_PORT"])
COORD_URL = f"http://127.0.0.1:{COORD_PORT}"


# ---------------------------------------------------------------------------
# Prometheus exporter — polls coordinator REST API, serves /metrics
# ---------------------------------------------------------------------------

def _fetch_json(path: str):
    try:
        with urllib.request.urlopen(COORD_URL + path, timeout=5) as r:
            import json
            return json.loads(r.read())
    except Exception as exc:
        log.debug("fetch %s failed: %s", path, exc)
        return None


def collect_metrics() -> str:
    t0 = time.monotonic()
    lines: list[str] = []

    def gauge(name, value, labels=None, help_text=""):
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        ls = ""
        if labels:
            ls = "{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}"
        lines.append(f"{name}{ls} {value}")

    healthz = _fetch_json("/healthz")
    up = 1.0 if (healthz and healthz.get("status") == "healthy") else 0.0
    gauge("lmcache_coordinator_up", up,
          help_text="1 if the LMCache MP coordinator /healthz reports healthy")

    if up < 1.0:
        gauge("lmcache_coordinator_scrape_duration_seconds",
              time.monotonic() - t0,
              help_text="Time in seconds to scrape the LMCache coordinator")
        return "\n".join(lines) + "\n"

    instances_resp = _fetch_json("/instances")
    instances = (instances_resp or {}).get("instances", [])
    lines.append("# HELP lmcache_coordinator_instances_registered Number of MP servers currently registered with the coordinator")
    lines.append("# TYPE lmcache_coordinator_instances_registered gauge")
    lines.append(f"lmcache_coordinator_instances_registered {len(instances)}")

    if instances:
        lines.append("# HELP lmcache_coordinator_instance_info Non-zero for each registered instance; labels carry metadata")
        lines.append("# TYPE lmcache_coordinator_instance_info gauge")
        for inst in instances:
            iid = inst.get("instance_id", "")
            ip = inst.get("ip", "")
            port = str(inst.get("http_port", ""))
            lines.append(f'lmcache_coordinator_instance_info{{instance_id="{iid}",ip="{ip}",http_port="{port}"}} 1')

    dir_stats = _fetch_json("/directory/stats")
    if dir_stats is not None:
        gauge("lmcache_coordinator_directory_keys_total",
              dir_stats.get("num_keys", 0),
              help_text="Keys with at least one placement in the key directory")
        gauge("lmcache_coordinator_directory_placements_total",
              dir_stats.get("num_placements", 0),
              help_text="Total placements across all tracked keys")
        inst_stats = dir_stats.get("instances", {})
        if inst_stats:
            lines.append("# HELP lmcache_coordinator_instance_l1_keys L1 keys reported per instance")
            lines.append("# TYPE lmcache_coordinator_instance_l1_keys gauge")
            lines.append("# HELP lmcache_coordinator_instance_seq_gap 1 if an event-stream gap was detected")
            lines.append("# TYPE lmcache_coordinator_instance_seq_gap gauge")
            for iid, s in inst_stats.items():
                lines.append(f'lmcache_coordinator_instance_l1_keys{{instance_id="{iid}"}} {s.get("num_l1_keys", 0)}')
                lines.append(f'lmcache_coordinator_instance_seq_gap{{instance_id="{iid}"}} {1 if s.get("gap_detected") else 0}')

    quota_list = _fetch_json("/quota?tier=l2")
    if quota_list is not None:
        # API returns {total_gb, by_cache_salt: [{cache_salt, usage_gb, quota_limit_gb, quota_exists}]}
        GB = 1024 ** 3
        total_gb = quota_list.get("total_gb", 0)
        lines.append("# HELP lmcache_coordinator_quota_usage_bytes_total Total L2 usage bytes tracked by coordinator")
        lines.append("# TYPE lmcache_coordinator_quota_usage_bytes_total gauge")
        lines.append(f"lmcache_coordinator_quota_usage_bytes_total {total_gb * GB:.0f}")
        entries = quota_list.get("by_cache_salt", [])
        if entries:
            lines.append("# HELP lmcache_coordinator_quota_usage_bytes L2 byte usage per cache_salt")
            lines.append("# TYPE lmcache_coordinator_quota_usage_bytes gauge")
            lines.append("# HELP lmcache_coordinator_quota_limit_bytes L2 quota limit per cache_salt; -1 = unlimited")
            lines.append("# TYPE lmcache_coordinator_quota_limit_bytes gauge")
            for entry in entries:
                salt = entry.get("cache_salt", "")
                usage = entry.get("usage_gb", 0) * GB
                limit = entry.get("quota_limit_gb", -1)
                limit_bytes = limit * GB if (limit and limit > 0) else -1
                lines.append(f'lmcache_coordinator_quota_usage_bytes{{cache_salt="{salt}"}} {usage:.0f}')
                lines.append(f'lmcache_coordinator_quota_limit_bytes{{cache_salt="{salt}"}} {limit_bytes:.0f}')

    gauge("lmcache_coordinator_scrape_duration_seconds",
          time.monotonic() - t0,
          help_text="Time in seconds to scrape the LMCache coordinator REST API")

    return "\n".join(lines) + "\n"


class MetricsHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path in ("/metrics", "/metrics/"):
            body = collect_metrics().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path in ("/healthz", "/health"):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass


def run_metrics_server():
    server = http.server.HTTPServer(("0.0.0.0", METRICS_PORT), MetricsHandler)
    log.info("Prometheus exporter listening on :%d/metrics", METRICS_PORT)
    server.serve_forever()


# ---------------------------------------------------------------------------
# Coordinator — run via uvicorn in this process
# ---------------------------------------------------------------------------

def run_coordinator():
    from lmcache.v1.mp_coordinator.app import create_app
    from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
    import uvicorn

    cfg = MPCoordinatorConfig(
        host="0.0.0.0",
        port=COORD_PORT,
        instance_timeout=float(os.environ["LMCACHE_COORDINATOR_INSTANCE_TIMEOUT"]),
        health_check_interval=float(os.environ["LMCACHE_COORDINATOR_HEALTH_INTERVAL"]),
        eviction_check_interval=float(os.environ["LMCACHE_COORDINATOR_EVICTION_INTERVAL"]),
        chunk_size=int(os.environ["LMCACHE_CHUNK_SIZE"]),
        timeout_keep_alive=int(os.environ["LMCACHE_COORDINATOR_KEEPALIVE"]),
    )
    app = create_app(cfg)
    log.info("Starting LMCache coordinator on :%d", COORD_PORT)
    uvicorn.run(
        app,
        host=cfg.host,
        port=cfg.port,
        timeout_keep_alive=cfg.timeout_keep_alive,
        log_level="warning",
    )


if __name__ == "__main__":
    if os.environ.get("LMCACHE_DISABLE_BANNER") != "1":
        pass  # banner suppressed via env in compose

    # Start the Prometheus exporter in a daemon thread
    t = threading.Thread(target=run_metrics_server, daemon=True, name="metrics")
    t.start()

    # Run the coordinator in the main thread (blocks until shutdown)
    try:
        run_coordinator()
    except KeyboardInterrupt:
        log.info("Shutting down")
        sys.exit(0)
