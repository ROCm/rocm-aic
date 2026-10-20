#!/usr/bin/env python3
"""AIC L2 dynamic-file monitor.

Watches the nixl_posix_dynamic L2 directory via inotify, counting per-file
read-back events (IN_OPEN) and periodically writing Prometheus text-format
metrics to a textfile that node_exporter picks up.

Metrics emitted:
  aic_l2_dynamic_files_total       - current file count in L2 dir
  aic_l2_dynamic_bytes_total       - total bytes used by L2 files
  aic_l2_dynamic_file_reads_total  - cumulative IN_OPEN events (reads)
  aic_l2_dynamic_read_count_bucket - histogram: distribution of per-file
                                     read counts across all L2 files
"""
import os, sys, time, threading, collections, struct, fcntl, ctypes

L2_DIR      = os.environ.get("AIC_L2_DIR", "/data/nvme/lmcache")
PROM_FILE   = os.environ.get("AIC_PROM_FILE", "/prom/aic_l2_monitor.prom")
INTERVAL    = int(os.environ.get("AIC_PROM_INTERVAL", "15"))

# Per-file read counters (filename → count)
_read_counts: dict[str, int] = collections.defaultdict(int)
_lock = threading.Lock()
_total_reads = 0

# ── inotify constants ─────────────────────────────────────────────────────────
IN_OPEN        = 0x00000020
IN_CREATE      = 0x00000100
IN_DELETE      = 0x00000200
IN_MOVED_TO    = 0x00000080
_INOTIFY_EVENT = struct.Struct("iIII")

def _inotify_init():
    return ctypes.CDLL("libc.so.6", use_errno=True).inotify_init()

def _inotify_add_watch(fd, path, mask):
    return ctypes.CDLL("libc.so.6", use_errno=True).inotify_add_watch(
        fd, path.encode(), mask)

def _inotify_watcher():
    global _total_reads
    fd = _inotify_init()
    if fd < 0:
        print("inotify_init failed — file read tracking unavailable", file=sys.stderr)
        return
    mask = IN_OPEN | IN_CREATE | IN_DELETE | IN_MOVED_TO
    wd = _inotify_add_watch(fd, L2_DIR, mask)
    if wd < 0:
        print(f"inotify_add_watch failed on {L2_DIR}", file=sys.stderr)
        return

    buf = bytearray(4096)
    while True:
        try:
            n = os.read(fd, len(buf))
        except Exception:
            break
        offset = 0
        while offset < len(n):
            ev_wd, ev_mask, ev_cookie, ev_len = _INOTIFY_EVENT.unpack_from(n, offset)
            offset += _INOTIFY_EVENT.size
            name = b""
            if ev_len > 0:
                name = bytes(n[offset:offset + ev_len]).rstrip(b"\x00")
                offset += ev_len
            fname = name.decode(errors="replace")
            if ev_mask & IN_OPEN and fname.endswith(".bin"):
                with _lock:
                    _read_counts[fname] += 1
                    _total_reads += 1

def _write_prom():
    while True:
        time.sleep(INTERVAL)
        try:
            # File/size stats
            files = [f for f in os.listdir(L2_DIR) if f.endswith(".bin")]
            total_files = len(files)
            total_bytes = sum(
                os.path.getsize(os.path.join(L2_DIR, f))
                for f in files
                if os.path.exists(os.path.join(L2_DIR, f))
            )

            with _lock:
                counts = dict(_read_counts)
                total_r = _total_reads

            # Build histogram of per-file read counts
            buckets = [0, 1, 2, 5, 10, 20, 50, 100, float("inf")]
            hist = collections.Counter()
            for v in counts.values():
                for b in buckets:
                    if v <= b:
                        hist[b] += 1
                        break
            # Files never opened → not in counts dict
            never_read = total_files - len(counts)

            # Non-cumulative per-bucket counts: how many files have been read
            # exactly in this range (not ≤N). Easier to read as a bar chart.
            bucket_labels = ["0", "1", "2", "5", "10", "20", "50", "100", ">100"]
            bucket_counts = [never_read] + [hist.get(b, 0) for b in buckets[1:-1]] + [hist.get(float("inf"), 0)]

            lines = [
                "# HELP aic_l2_dynamic_files_total Current number of KV chunk files in the L2 NVMe directory.",
                "# TYPE aic_l2_dynamic_files_total gauge",
                f"aic_l2_dynamic_files_total {total_files}",
                "# HELP aic_l2_dynamic_bytes_total Total bytes used by L2 KV chunk files.",
                "# TYPE aic_l2_dynamic_bytes_total gauge",
                f"aic_l2_dynamic_bytes_total {total_bytes}",
                "# HELP aic_l2_dynamic_file_reads_total Cumulative IN_OPEN events on L2 chunk files since monitor start.",
                "# TYPE aic_l2_dynamic_file_reads_total counter",
                f"aic_l2_dynamic_file_reads_total {total_r}",
                "# HELP aic_l2_dynamic_read_count Files per read-count range (non-cumulative): how many NVMe files have been read exactly N times.",
                "# TYPE aic_l2_dynamic_read_count gauge",
            ]
            for label, count in zip(bucket_labels, bucket_counts):
                lines.append(f'aic_l2_dynamic_read_count{{reads="{label}"}} {count}')

            os.makedirs(os.path.dirname(PROM_FILE), exist_ok=True)
            tmp = PROM_FILE + ".tmp"
            with open(tmp, "w") as f:
                f.write("\n".join(lines) + "\n")
            os.replace(tmp, PROM_FILE)

        except Exception as e:
            print(f"prom write error: {e}", file=sys.stderr)

if __name__ == "__main__":
    os.makedirs(L2_DIR, exist_ok=True)
    print(f"Monitoring {L2_DIR} → {PROM_FILE} (interval={INTERVAL}s)")
    threading.Thread(target=_inotify_watcher, daemon=True).start()
    _write_prom()  # blocks forever
