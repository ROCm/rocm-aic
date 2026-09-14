#!/usr/bin/env bash
set -euo pipefail

# Garbage-collect a BuildKit `type=local` cache directory.
#
# The local cache exporter is append-only: every `--cache-to type=local,dest=DIR`
# writes a fresh manifest plus its blobs under DIR/blobs/sha256 and rewrites
# DIR/index.json to point at that one manifest.  Blobs belonging to the
# *previous* manifest are left on disk forever.  `--cache-from type=local,src=DIR`
# only ever reads what index.json reaches, so those leftovers speed up nothing --
# they are pure occupancy on shared NFS.  (Measured on SPUR 2026-09-14: 233 GB of
# cache, 45 GB reachable, 188 GB orphaned across two arch dirs.)
#
# This removes exactly the unreachable blobs.  Reachable blobs are never touched
# regardless of age, so a prune can never cost a cache hit.
#
# Usage:
#   prune-local-buildcache.sh [--dry-run] [--grace-minutes N] <cache-dir> [...]
#
# A cache-dir here is the per-arch leaf that BuildKit was pointed at (the dir
# holding oci-layout/index.json/blobs), or the parent holding several of them --
# both are accepted.
#
# --grace-minutes N (default 1440) spares any unreachable blob modified within
# the last N minutes.  A concurrent build writes its blobs *before* it rewrites
# index.json, so during that window its blobs look unreachable; deleting them
# would corrupt the cache it is about to publish.  N must therefore exceed the
# longest build (AIC_BUILD_TIME defaults to 2h), hence the 24h default.

GRACE_MINUTES=1440
DRY_RUN=0
declare -a DIRS=()

log() { printf '[prune-buildcache] %s\n' "$*" >&2; }
die() { printf '[prune-buildcache] ERROR: %s\n' "$*" >&2; exit 1; }

while (($#)); do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --grace-minutes)
            [[ "${2:-}" =~ ^[0-9]+$ ]] || die "--grace-minutes needs a non-negative integer"
            GRACE_MINUTES="$2"; shift 2 ;;
        -h | --help)
            sed -n '5,30p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        -*) die "unknown option: $1" ;;
        *) DIRS+=("$1"); shift ;;
    esac
done
((${#DIRS[@]})) || die "usage: $0 [--dry-run] [--grace-minutes N] <cache-dir> [...]"

command -v python3 >/dev/null 2>&1 || die "python3 is required"

# Expand a parent dir into the OCI layouts underneath it, so callers can pass
# either the per-arch leaf or the AIC_CACHE_DIR root.
declare -a LAYOUTS=()
for d in "${DIRS[@]}"; do
    if [[ -f "${d}/oci-layout" ]]; then
        LAYOUTS+=("${d}")
    elif [[ -d "${d}" ]]; then
        for sub in "${d}"/*/; do
            [[ -f "${sub}oci-layout" ]] && LAYOUTS+=("${sub%/}")
        done
    else
        log "skipping ${d}: not a directory"
    fi
done
((${#LAYOUTS[@]})) || { log "no OCI cache layouts found under: ${DIRS[*]}"; exit 0; }

log "grace ${GRACE_MINUTES}m, dry-run=${DRY_RUN}, layouts: ${#LAYOUTS[@]}"

# The reachability walk is JSON work, so it lives in python rather than being
# hand-rolled over the manifests with grep.  It prints a per-layout report and
# deletes; the exit status propagates through set -e.
AIC_PRUNE_GRACE_MINUTES="${GRACE_MINUTES}" AIC_PRUNE_DRY_RUN="${DRY_RUN}" \
    python3 - "${LAYOUTS[@]}" <<'PYEOF'
import json
import os
import sys
import time

GRACE = int(os.environ["AIC_PRUNE_GRACE_MINUTES"]) * 60
DRY_RUN = os.environ["AIC_PRUNE_DRY_RUN"] == "1"
GIB = 1 << 30
NOW = time.time()


def digest_path(blob_dir, digest):
    """Map an OCI 'sha256:<hex>' digest onto its file, rejecting path escapes."""
    algo, _, hexd = digest.partition(":")
    if algo != "sha256" or not hexd.isalnum():
        return None
    return os.path.join(blob_dir, hexd)


def reachable(blob_dir, index):
    """Every blob transitively referenced by index.json.

    mode=max caches nest an image index above the cache manifest in some
    BuildKit versions, so descriptors are walked rather than assumed flat.
    """
    seen, pending = set(), [m["digest"] for m in index.get("manifests", [])]
    while pending:
        digest = pending.pop()
        if digest in seen:
            continue
        seen.add(digest)
        path = digest_path(blob_dir, digest)
        if path is None or not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as fh:
                doc = json.loads(fh.read())
        except (ValueError, OSError):
            continue  # a layer blob, not JSON -- nothing further to follow
        if not isinstance(doc, dict):
            continue
        for desc in doc.get("manifests", []) + doc.get("layers", []):
            if isinstance(desc, dict) and "digest" in desc:
                pending.append(desc["digest"])
        config = doc.get("config")
        if isinstance(config, dict) and "digest" in config:
            pending.append(config["digest"])
    return seen


total_freed = total_kept = 0
failures = []

for layout in sys.argv[1:]:
    blob_dir = os.path.join(layout, "blobs", "sha256")
    index_path = os.path.join(layout, "index.json")
    if not os.path.isdir(blob_dir) or not os.path.isfile(index_path):
        print("[prune-buildcache] skipping %s: not a BuildKit local cache" % layout, file=sys.stderr)
        continue
    try:
        with open(index_path, "rb") as fh:
            index = json.loads(fh.read())
    except (ValueError, OSError) as exc:
        # Never delete on an unreadable index: everything would look unreachable.
        failures.append("%s: cannot read index.json (%s)" % (layout, exc))
        continue

    keep = {os.path.basename(p) for p in
            (digest_path(blob_dir, d) for d in reachable(blob_dir, index)) if p}
    freed = kept = skipped_young = 0
    removed = young = 0

    for name in os.listdir(blob_dir):
        path = os.path.join(blob_dir, name)
        try:
            st = os.stat(path)
        except OSError:
            continue
        if not os.path.isfile(path):
            continue
        if name in keep:
            kept += st.st_size
            continue
        if NOW - st.st_mtime < GRACE:
            skipped_young += st.st_size
            young += 1
            continue
        if DRY_RUN:
            freed += st.st_size
            removed += 1
            continue
        try:
            os.unlink(path)
        except OSError as exc:
            failures.append("%s: cannot remove %s (%s)" % (layout, name, exc))
            continue
        freed += st.st_size
        removed += 1

    print("[prune-buildcache] %s" % layout, file=sys.stderr)
    print("[prune-buildcache]   reachable %7.1f GiB | %s %d orphan blobs %7.1f GiB | "
          "in grace %d %7.1f GiB"
          % (kept / GIB, "would remove" if DRY_RUN else "removed", removed,
             freed / GIB, young, skipped_young / GIB), file=sys.stderr)
    total_freed += freed
    total_kept += kept

print("[prune-buildcache] total: %s %.1f GiB, kept %.1f GiB reachable"
      % ("would free" if DRY_RUN else "freed", total_freed / GIB, total_kept / GIB),
      file=sys.stderr)

if failures:
    for msg in failures:
        print("[prune-buildcache] WARNING: %s" % msg, file=sys.stderr)
PYEOF

# BuildKit probes each cache dir for fsverity support and leaves the empty
# .fsverity-check-<rand> dir behind on every export (32 of them had accumulated
# on SPUR).  Trivial in bytes, but they make the cache dir unreadable by hand.
# rmdir (not rm -rf) so a probe dir that unexpectedly holds files is reported
# rather than deleted.
for layout in "${LAYOUTS[@]}"; do
    stale=0
    while IFS= read -r -d '' probe; do
        stale=$((stale + 1))
        if ((DRY_RUN == 0)); then
            rmdir "${probe}" || log "WARNING: could not remove probe dir ${probe}"
        fi
    done < <(find "${layout}" -maxdepth 1 -type d -name '.fsverity-check-*' \
        -mmin "+${GRACE_MINUTES}" -print0)
    if ((stale > 0)); then
        if ((DRY_RUN)); then
            log "${layout}: would remove ${stale} stale .fsverity-check-* dirs"
        else
            log "${layout}: removed ${stale} stale .fsverity-check-* dirs"
        fi
    fi
done
