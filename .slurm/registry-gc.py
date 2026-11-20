#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""Choose and delete the manifests in a registry storage that nothing still uses.

Run by aic_registry_gc (.slurm/image-dist-lib.sh), which holds the storage's
exclusive lock, so no job serves the storage while this reads or changes it.
This only deletes manifests, through the registry API; `registry
garbage-collect` then removes the blobs no remaining manifest references.

A manifest is kept when any of these is RETAIN_DAYS old or younger, or when a
manifest kept for that reason lists it (an index keeps its platform and
attestation manifests):
  * its revision link: when it was last pushed by digest
  * the current link of a tag naming it: when the tag last moved to it
  * a stamp aic_registry_stop wrote: when a job last read or wrote it, by tag
    or by digest.  A push of something the registry already holds sends only
    a HEAD, so the link times alone miss a build that reused its cache whole.
A manifest is also kept, however old, when a .ref under REF_DIRS names it:
some job may still fetch it.

Exit status: 0 also when there is nothing to do; 1 on an error, after which the
caller must not run garbage-collect.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

STAMP_DIR = "aic-gc/used"
# Written by the first aic_registry_stop that records a stamp; until it is
# RETAIN_DAYS old, reads before it are unrecorded and nothing is deleted.
STAMP_EPOCH = "aic-gc/stamping-since"
V2 = Path("docker/registry/v2")
SKIP_DIRS = {"_manifests", "_layers", "_uploads"}


def log(msg: str) -> None:
    print(f"[registry-gc] {msg}", flush=True)


def mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def read_link(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return None


def find_repos(root: Path) -> list[str]:
    """Repository names: every directory under root holding _manifests."""
    repos = []
    for dirpath, dirnames, _ in os.walk(root):
        if "_manifests" in dirnames:
            repos.append(os.path.relpath(dirpath, root))
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    return sorted(repos)


class Repo:
    def __init__(self, path: Path):
        self.revisions: dict[str, float] = {}  # digest -> link mtime
        self.tags: dict[str, tuple[str, float]] = {}  # tag -> (digest, mtime)
        rev = path / "_manifests/revisions/sha256"
        if rev.is_dir():
            for d in rev.iterdir():
                t = mtime(d / "link")
                if t is not None:
                    self.revisions[f"sha256:{d.name}"] = t
        tags = path / "_manifests/tags"
        if tags.is_dir():
            for d in tags.iterdir():
                link = d / "current/link"
                digest, t = read_link(link), mtime(link)
                if digest and t is not None:
                    self.tags[d.name] = (digest, t)


def read_stamps(storage: Path) -> dict[tuple[str, str], float]:
    """(repo, tag-or-digest) -> last time a job used it."""
    stamps = {}
    d = storage / STAMP_DIR
    if d.is_dir():
        for f in d.iterdir():
            repo, sep, ref = urllib.parse.unquote(f.name).rpartition("@")
            t = mtime(f)
            if sep and repo and ref and t is not None:
                stamps[(repo, ref)] = t
    return stamps


def read_refs(storage: Path, ref_dirs: list[str]) -> set[tuple[str, str]]:
    """(repo, digest) of every .ref under ref_dirs that points at storage."""
    want = os.path.realpath(storage)
    named = set()
    for top in ref_dirs:
        for dirpath, _, files in os.walk(top):
            for name in files:
                if not name.endswith(".ref"):
                    continue
                try:
                    fields = (Path(dirpath) / name).read_text().split()
                except OSError as e:
                    log(f"WARN: cannot read {dirpath}/{name}: {e}")
                    continue
                if len(fields) < 2 or "@" not in fields[1]:
                    continue
                if os.path.realpath(fields[0]) != want:
                    continue
                repo, _, digest = fields[1].partition("@")
                named.add((repo, digest))
    return named


def blob_path(v2: Path, digest: str) -> Path:
    algo, _, hexd = digest.partition(":")
    return v2 / "blobs" / algo / hexd[:2] / hexd / "data"


def load_manifest(v2: Path, digest: str) -> dict:
    try:
        return json.loads(blob_path(v2, digest).read_bytes())
    except (OSError, ValueError):
        return {}


def blob_size(v2: Path, digest: str) -> int:
    try:
        return blob_path(v2, digest).stat().st_size
    except OSError:
        return 0


def plan(storage: Path, retain_days: float, ref_dirs: list[str], now: float):
    v2 = storage / V2
    cutoff = now - retain_days * 86400
    stamps = read_stamps(storage)
    named = read_refs(storage, ref_dirs)
    repos = {name: Repo(v2 / "repositories" / name) for name in find_repos(v2 / "repositories")}

    keep: dict[str, set[str]] = {}
    drop: dict[str, set[str]] = {}
    for name, repo in repos.items():
        roots = set()
        for digest, t in repo.revisions.items():
            if t >= cutoff or stamps.get((name, digest), 0) >= cutoff or (name, digest) in named:
                roots.add(digest)
        for tag, (digest, t) in repo.tags.items():
            if t >= cutoff or stamps.get((name, tag), 0) >= cutoff:
                roots.add(digest)
        # An index keeps what it lists.  A cache index lists layers, which are
        # not revisions and are left to garbage-collect.
        kept, todo = set(), list(roots)
        while todo:
            digest = todo.pop()
            if digest in kept:
                continue
            kept.add(digest)
            for child in load_manifest(v2, digest).get("manifests", []):
                c = child.get("digest", "")
                if c in repo.revisions:
                    todo.append(c)
        keep[name] = kept
        drop[name] = set(repo.revisions) - kept
    return repos, keep, drop


def referenced_blobs(v2: Path, keep: dict[str, set[str]]) -> set[str]:
    """Blobs the kept manifests reach: what garbage-collect will mark."""
    marked = set()
    for digests in keep.values():
        for digest in digests:
            marked.add(digest)
            m = load_manifest(v2, digest)
            if "config" in m:
                marked.add(m["config"].get("digest", ""))
            for key in ("layers", "manifests", "blobs"):
                for d in m.get(key, []):
                    marked.add(d.get("digest", ""))
    return marked


def all_blobs(v2: Path) -> dict[str, int]:
    out = {}
    root = v2 / "blobs/sha256"
    if root.is_dir():
        for prefix in root.iterdir():
            for d in prefix.iterdir():
                out[f"sha256:{d.name}"] = blob_size(v2, f"sha256:{d.name}")
    return out


def delete(registry: str, repo: str, digest: str) -> bool:
    url = f"{registry}/v2/{repo}/manifests/{digest}"
    req = urllib.request.Request(url, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status == 202
    except urllib.error.HTTPError as e:
        if e.code == 404:  # already gone
            return True
        log(f"ERROR: DELETE {repo}@{digest}: HTTP {e.code} {e.read()[:200]!r}")
    except urllib.error.URLError as e:
        log(f"ERROR: DELETE {repo}@{digest}: {e.reason}")
    return False


def prune_stamps(storage: Path, cutoff: float) -> int:
    n = 0
    d = storage / STAMP_DIR
    if d.is_dir():
        for f in d.iterdir():
            t = mtime(f)
            if t is not None and t < cutoff:
                f.unlink(missing_ok=True)
                n += 1
    return n


def gib(n: int) -> str:
    return f"{n / 2**30:.1f} GiB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("storage", type=Path, help="registry storage (the directory served as /var/lib/registry)")
    ap.add_argument("--retain-days", type=float, required=True)
    ap.add_argument("--ref-dir", dest="ref_dirs", action="append", default=[], help="tree to search for .ref files; repeatable")
    ap.add_argument("--registry", help="URL of a registry serving STORAGE with deletes enabled; omit for a dry run")
    args = ap.parse_args()

    storage = args.storage.resolve()
    if not (storage / V2 / "repositories").is_dir():
        log(f"{storage} holds no registry repositories; nothing to do")
        return 0
    now = time.time()
    cutoff = now - args.retain_days * 86400
    epoch = mtime(storage / STAMP_EPOCH)
    warm = epoch is not None and epoch <= cutoff
    if not warm:
        since = time.strftime("%Y-%m-%d %H:%M", time.localtime(epoch)) if epoch else "never"
        log(f"usage stamps recorded since {since}, under {args.retain_days:g} days: "
            "reads before then are unknown, so this run deletes nothing")

    repos, keep, drop = plan(storage, args.retain_days, args.ref_dirs, now)
    v2 = storage / V2
    blobs = all_blobs(v2)
    marked = referenced_blobs(v2, keep)
    freed = sum(size for d, size in blobs.items() if d not in marked)
    total_drop = sum(len(v) for v in drop.values())
    for name in sorted(repos):
        if drop[name]:
            log(f"  {name}: delete {len(drop[name])} of {len(repos[name].revisions)} manifests")
    log(f"{len(repos)} repositories; keep {sum(len(v) for v in keep.values())} manifests, "
        f"delete {total_drop}; blobs {len(blobs)} ({gib(sum(blobs.values()))}), "
        f"about {gib(freed)} unreferenced after this")

    if not args.registry or not warm:
        if not args.registry:
            log("dry run: nothing deleted")
        return 0
    failed = 0
    for name in sorted(drop):
        for digest in sorted(drop[name]):
            if not delete(args.registry, name, digest):
                failed += 1
    if failed:
        log(f"ERROR: {failed} of {total_drop} deletes failed")
        return 1
    log(f"deleted {total_drop} manifests; removed {prune_stamps(storage, cutoff)} expired stamps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
