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
import os, signal, sys, time, threading, collections, struct, ctypes

L2_DIR      = os.environ["AIC_L2_DIR"]
PROM_FILE   = os.environ["AIC_PROM_FILE"]
INTERVAL    = int(os.environ["AIC_PROM_INTERVAL"])

_lock = threading.Lock()

# Per-file read counters (filename → count), updated by inotify IN_OPEN events
_read_counts: dict[str, int] = collections.defaultdict(int)
_total_reads: int = 0

# File count and byte total maintained incrementally; initialised by _initial_scan()
_file_sizes: dict[str, int] = {}  # fname → size in bytes
_file_count: int = 0
_total_bytes: int = 0

# inotify constants
IN_OPEN       = 0x00000020
IN_CREATE     = 0x00000100
IN_DELETE     = 0x00000200
IN_MOVED_FROM = 0x00000040
IN_MOVED_TO   = 0x00000080
_INOTIFY_EVENT = struct.Struct("iIII")

_libc = ctypes.CDLL("libc.so.6", use_errno=True)


def _inotify_watcher():
    global _total_reads, _file_count, _total_bytes
    fd = _libc.inotify_init()
    if fd < 0:
        print("inotify_init failed", file=sys.stderr)
        return
    mask = IN_OPEN | IN_CREATE | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO
    if _libc.inotify_add_watch(fd, L2_DIR.encode(), mask) < 0:
        print(f"inotify_add_watch failed on {L2_DIR}", file=sys.stderr)
        return

    buf = bytearray(65536)
    while True:
        try:
            n = os.read(fd, len(buf))
        except Exception:
            break
        offset = 0
        while offset < len(n):
            ev_wd, ev_mask, ev_cookie, ev_len = _INOTIFY_EVENT.unpack_from(n, offset)
            offset += _INOTIFY_EVENT.size
            fname = ""
            if ev_len > 0:
                fname = bytes(n[offset:offset + ev_len]).rstrip(b"\x00").decode(errors="replace")
                offset += ev_len
            if not fname.endswith(".bin"):
                continue
            if ev_mask & IN_OPEN:
                with _lock:
                    _read_counts[fname] += 1
                    _total_reads += 1
            elif ev_mask & (IN_CREATE | IN_MOVED_TO):
                # stat outside the lock to avoid blocking the watcher
                try:
                    sz = os.path.getsize(os.path.join(L2_DIR, fname))
                except OSError:
                    sz = 0
                with _lock:
                    if fname not in _file_sizes:
                        _file_count += 1
                    else:
                        _total_bytes -= _file_sizes[fname]
                    _file_sizes[fname] = sz
                    _total_bytes += sz
            elif ev_mask & (IN_DELETE | IN_MOVED_FROM):
                with _lock:
                    if fname in _file_sizes:
                        _total_bytes -= _file_sizes.pop(fname)
                        _file_count -= 1


def _initial_scan():
    """One-time scan at startup. Inotify watch must be armed first so no
    events are missed during the scan. Uses scandir for a single syscall
    per entry rather than listdir + getsize per file."""
    global _file_count, _total_bytes
    try:
        with os.scandir(L2_DIR) as it:
            entries = [(e.name, e.stat().st_size) for e in it if e.name.endswith(".bin")]
    except OSError as e:
        print(f"initial scan failed: {e}", file=sys.stderr)
        return
    with _lock:
        for name, sz in entries:
            if name not in _file_sizes:  # skip files inotify already registered
                _file_sizes[name] = sz
                _total_bytes += sz
        _file_count = len(_file_sizes)
    print(f"initial scan: {_file_count} files, {_total_bytes >> 20} MiB", file=sys.stderr)


def _write_prom():
    while True:
        time.sleep(INTERVAL)
        try:
            with _lock:
                total_files = _file_count
                total_bytes = _total_bytes
                counts = dict(_read_counts)
                total_r = _total_reads

            # Build histogram of per-file read counts (pure in-memory, no I/O)
            buckets = [0, 1, 2, 5, 10, 20, 50, 100, float("inf")]
            hist = collections.Counter()
            for v in counts.values():
                for b in buckets:
                    if v <= b:
                        hist[b] += 1
                        break
            never_read = total_files - len(counts)

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


def _wait_for_l2_dir():
    """Block until the L2 directory exists.

    The lmcache server creates it when it sets up its L2 pool, which on a
    fresh pool is after this monitor starts.  It is not created here: the
    pool is mounted read-only, so makedirs raised EROFS and the monitor
    exited for the life of the stack.  With no L2 configured the directory
    never appears and the monitor reports nothing."""
    if os.path.isdir(L2_DIR):
        return
    print(f"waiting for {L2_DIR} to be created", file=sys.stderr)
    # The textfile volume outlives the stack: without this, node-exporter
    # serves the previous run's numbers for as long as the wait lasts.
    try:
        os.remove(PROM_FILE)
    except OSError:
        pass
    while not os.path.isdir(L2_DIR):
        time.sleep(1)


if __name__ == "__main__":
    # PID 1 ignores SIGTERM unless it handles it: `docker stop` would wait out
    # its 10 s grace period, which the wait below can now last forever.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    _wait_for_l2_dir()
    print(f"Monitoring {L2_DIR} → {PROM_FILE} (interval={INTERVAL}s)", file=sys.stderr)
    # Arm inotify before scanning so events during the scan are not lost
    threading.Thread(target=_inotify_watcher, daemon=True).start()
    _initial_scan()
    _write_prom()  # blocks forever
