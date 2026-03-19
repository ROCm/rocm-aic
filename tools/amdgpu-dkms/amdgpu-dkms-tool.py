#!/usr/bin/env python3
"""
Repack a .deb: extract, optionally apply patches, bump version
in DEBIAN/control, DEBIAN/postinst, DEBIAN/prerm, every
dkms.conf, and the usr/src/ directory name, then build a new
.deb. This lets DKMS treat the repacked module as a distinct
version that can coexist with the original.

Input may be a local path or a URL (http/https). When given a URL, the .deb is
downloaded to debs/ by default; use --download-dir to save elsewhere.

Example (local file):
  ./amdgpu-dkms-tool.py debs/amdgpu-dkms_6.18.8.31200000-2295296.24.04_all.deb \\
    --suffix +local1 --patch ./patches --output amdgpu-dkms_..._all.deb

Example (URL; downloads to debs/ then repacks):
  ./amdgpu-dkms-tool.py <artifactory-url-to-amdgpu-dkms.deb> \\
    --version 6.18.8.31200000-2295296.24.05 --patch ./patches --output out.deb

Patches are applied from inside the single usr/src/amdgpu-<label>/ directory,
so use paths relative to that (e.g. amd/dkms/pre-build.sh). They are applied
with patch -p1 and do not need to change when repacking a different .deb
version.
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlparse
from urllib.request import urlretrieve

def main():
    parser = argparse.ArgumentParser(
        description="Repack a .deb with optional patches and version bump.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_deb",
        metavar="input.deb",
        help="Path or URL of the input .deb file",
    )
    parser.add_argument(
        "--patch",
        metavar="PATH",
        help="Patch file or directory of .patch files (top-level only, sorted)",
    )
    parser.add_argument(
        "--version",
        metavar="NEW_VERSION",
        help="New version string (e.g. 1:6.18.8.31200000-2295296.24.05+local1)",
    )
    parser.add_argument(
        "--suffix",
        metavar="SUFFIX",
        help="Append to current version (e.g. +local1 or .1). Ignored if --version set",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Extract here instead of a temp dir; script does not delete it",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Do not remove extraction dir when using a temp dir",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Output .deb path; default: <Package>_<version>_<Arch>.deb in cwd",
    )
    parser.add_argument(
        "--download-dir",
        metavar="DIR",
        default=None,
        help="When input is a URL, save .deb here (default: debs). Created if missing",
    )
    args = parser.parse_args()

    if not args.version and not args.suffix:
        parser.error("One of --version or --suffix is required")

    parsed = urlparse(args.input_deb)
    if parsed.scheme in ("http", "https"):
        url = args.input_deb
        download_dir = os.path.abspath(args.download_dir or "debs")
        os.makedirs(download_dir, exist_ok=True)
        name = os.path.basename(parsed.path) or "downloaded.deb"
        local_path = os.path.join(download_dir, name)
        print(f"Downloading {url} -> {local_path}", flush=True)
        try:
            urlretrieve(url, local_path)
        except OSError as e:
            sys.stderr.write(f"Error: download failed: {e}\n")
            return 2
        input_deb = local_path
    else:
        input_deb = os.path.abspath(args.input_deb)
        if not os.path.isfile(input_deb):
            sys.stderr.write(f"Error: input file not found: {input_deb}\n")
            return 2

    used_temp = False
    if args.output_dir:
        extract_dir = os.path.abspath(args.output_dir)
        os.makedirs(extract_dir, exist_ok=True)
    else:
        extract_dir = tempfile.mkdtemp(prefix="amdgpu-dkms-repack.")
        used_temp = True

    try:
        # 1. Extract the .deb
        print(f"Extracting {input_deb} -> {extract_dir}", flush=True)
        ret = subprocess.run(
            ["dpkg-deb", "-R", input_deb, extract_dir],
            capture_output=True,
            text=True,
        )
        if ret.returncode != 0:
            sys.stderr.write(f"Error: dpkg-deb -R failed: {ret.stderr}")
            return 3

        control_path = os.path.join(extract_dir, "DEBIAN", "control")
        if not os.path.isfile(control_path):
            sys.stderr.write(f"Error: no DEBIAN/control in {extract_dir}\n")
            return 4

        # 2. Apply patches (from inside usr/src/amdgpu-<label>/ so paths stay
        #    version-agnostic, e.g. amd/dkms/pre-build.sh)
        if args.patch:
            patch_path = os.path.abspath(args.patch)
            if os.path.isfile(patch_path):
                patch_files = [patch_path]
            elif os.path.isdir(patch_path):
                patch_files = sorted(glob.glob(os.path.join(patch_path, "*.patch")))
            else:
                sys.stderr.write(f"Error: --patch is not a file or dir: {patch_path}\n")
                return 5
            usr_src = os.path.join(extract_dir, "usr", "src")
            amdgpu_dirs = [
                d for d in os.listdir(usr_src)
                if os.path.isdir(os.path.join(usr_src, d)) and d.startswith("amdgpu-")
            ] if os.path.isdir(usr_src) else []
            if len(amdgpu_dirs) != 1:
                sys.stderr.write(
                    f"Error: expected exactly one usr/src/amdgpu-* dir, "
                    f"got {len(amdgpu_dirs)}\n"
                )
                return 5
            patch_cwd = os.path.join(usr_src, amdgpu_dirs[0])
            for p in patch_files:
                print(f"Applying patch: {os.path.basename(p)}", flush=True)
                with open(p, "rb") as f:
                    ret = subprocess.run(
                        ["patch", "-p1"],
                        cwd=patch_cwd,
                        stdin=f,
                        capture_output=True,
                        text=True,
                    )
                if ret.returncode != 0:
                    sys.stderr.write(
                        f"Error: patch failed for {p}: {ret.stdout or ret.stderr}\n"
                    )
                    return 6

        # 3. Compute new version from DEBIAN/control
        with open(control_path, "r", encoding="utf-8") as f:
            control_content = f.read()
        version_match = re.search(
            r"^Version:\s*(.+)\s*$",
            control_content,
            re.MULTILINE,
        )
        if not version_match:
            sys.stderr.write(
                "Error: no Version line in DEBIAN/control\n"
            )
            return 7
        current_version = version_match.group(1).strip()
        if args.version:
            new_version = args.version
        else:
            new_version = current_version + args.suffix

        # 3a. Update DEBIAN/control
        control_content, n = re.subn(
            r"^(Version:\s*)" + re.escape(current_version)
            + r"(\s*)$",
            r"\g<1>" + new_version + r"\g<2>",
            control_content,
            count=1,
            flags=re.MULTILINE,
        )
        if not n:
            sys.stderr.write(
                "Error: failed to update Version in "
                "DEBIAN/control\n"
            )
            return 7
        with open(control_path, "w", encoding="utf-8") as f:
            f.write(control_content)
        print(
            f"DEBIAN/control: {current_version} -> {new_version}",
            flush=True,
        )

        # 3b. Locate the amdgpu source directory and read the
        #     DKMS version (may differ from the control version)
        usr_src = os.path.join(extract_dir, "usr", "src")
        amdgpu_dirs = [
            d for d in os.listdir(usr_src)
            if os.path.isdir(os.path.join(usr_src, d))
            and d.startswith("amdgpu-")
        ] if os.path.isdir(usr_src) else []
        if len(amdgpu_dirs) != 1:
            sys.stderr.write(
                "Error: expected exactly one "
                f"usr/src/amdgpu-* dir, got {len(amdgpu_dirs)}\n"
            )
            return 5
        old_src_dir = os.path.join(usr_src, amdgpu_dirs[0])
        old_dkms_ver = amdgpu_dirs[0].removeprefix("amdgpu-")

        # 3c. Update PACKAGE_VERSION in every dkms.conf
        for root, _dirs, files in os.walk(old_src_dir):
            for fname in files:
                if fname != "dkms.conf":
                    continue
                p = os.path.join(root, fname)
                with open(p, "r", encoding="utf-8") as f:
                    txt = f.read()
                m = re.search(
                    r'^PACKAGE_VERSION="([^"]+)"',
                    txt,
                    re.MULTILINE,
                )
                if not m:
                    continue
                txt = txt.replace(
                    f'PACKAGE_VERSION="{m.group(1)}"',
                    f'PACKAGE_VERSION="{new_version}"',
                )
                with open(p, "w", encoding="utf-8") as f:
                    f.write(txt)
                rel = os.path.relpath(p, extract_dir)
                print(
                    f"{rel}: {m.group(1)} -> {new_version}",
                    flush=True,
                )

        # 3d. Update CVERSION in DEBIAN/postinst
        postinst = os.path.join(
            extract_dir, "DEBIAN", "postinst"
        )
        if os.path.isfile(postinst):
            with open(postinst, "r", encoding="utf-8") as f:
                txt = f.read()
            txt, n = re.subn(
                r'^(CVERSION=).*$',
                rf'\g<1>{new_version}',
                txt,
                flags=re.MULTILINE,
            )
            if n:
                with open(postinst, "w", encoding="utf-8") as f:
                    f.write(txt)
                print(
                    f"DEBIAN/postinst CVERSION: "
                    f"{old_dkms_ver} -> {new_version}",
                    flush=True,
                )

        # 3e. Update VERSION in DEBIAN/prerm
        prerm = os.path.join(extract_dir, "DEBIAN", "prerm")
        if os.path.isfile(prerm):
            with open(prerm, "r", encoding="utf-8") as f:
                txt = f.read()
            txt, n = re.subn(
                r'^(VERSION=).*$',
                rf'\g<1>{new_version}',
                txt,
                flags=re.MULTILINE,
            )
            if n:
                with open(prerm, "w", encoding="utf-8") as f:
                    f.write(txt)
                print(
                    f"DEBIAN/prerm VERSION: "
                    f"{old_dkms_ver} -> {new_version}",
                    flush=True,
                )

        # 3f. Rename usr/src/amdgpu-<old> -> amdgpu-<new>
        new_src_dir = os.path.join(
            usr_src, f"amdgpu-{new_version}"
        )
        if old_src_dir != new_src_dir:
            os.rename(old_src_dir, new_src_dir)
            print(
                f"Renamed amdgpu-{old_dkms_ver} -> "
                f"amdgpu-{new_version}",
                flush=True,
            )

        # 4. Build the new .deb
        if args.output:
            out_deb = os.path.abspath(args.output)
        else:
            pkg_match = re.search(r"^Package:\s*(.+)\s*$", control_content, re.MULTILINE)
            arch_match = re.search(
                r"^Architecture:\s*(.+)\s*$", control_content, re.MULTILINE
            )
            if not pkg_match or not arch_match:
                sys.stderr.write(
                    "Error: Package or Architecture missing in DEBIAN/control\n"
                )
                return 8
            pkg = pkg_match.group(1).strip()
            arch = arch_match.group(1).strip()
            safe_version = new_version.replace("/", "%2f")
            out_deb = os.path.join(
                os.getcwd(), f"{pkg}_{safe_version}_{arch}.deb"
            )

        print(f"Repacking -> {out_deb}", flush=True)
        ret = subprocess.run(
            ["dpkg-deb", "--build", extract_dir, out_deb],
            capture_output=True,
            text=True,
        )
        if ret.returncode != 0:
            sys.stderr.write(f"Error: dpkg-deb --build failed: {ret.stderr}")
            return 9

        print(out_deb)
        return 0
    finally:
        if used_temp and not args.keep:
            shutil.rmtree(extract_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
