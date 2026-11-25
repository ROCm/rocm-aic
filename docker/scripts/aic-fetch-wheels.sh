#!/usr/bin/env bash
# -E inherit ERR trap in functions and subshells,
# -e exit the script upon commands failing,
# -u to error if the script ever references a variable that is unset,
# -o pipefail to error if any stage of a pipeline fails.
set -Eeuo pipefail

# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Download the pip wheels of one GitHub release, for `make build-from-wheels`.
#
#   aic-fetch-wheels.sh <owner/repo> <release-tag> <dest-dir>
#
# <dest-dir> ends up holding exactly the release's .whl assets.  A wheel already
# there with the published size and sha256 is kept; one the release no longer
# lists is REMOVED, because a rolling tag (nightly) republishes under new
# version strings and two vllm wheels in one directory fail the image build.
# Do not point it at a directory of wheels you built.
#
# The request is anonymous: a public release needs no more, and it works where
# the organisation refuses the caller's token.  GH_TOKEN or GITHUB_TOKEN is
# used only when the anonymous request fails (a private fork, a rate limit).

die() { printf 'aic-fetch-wheels: ERROR: %s\n' "$*" >&2; exit 1; }

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <owner/repo> <release-tag> <dest-dir>" >&2
    exit 2
fi
repo="$1"
tag="$2"
dest="$3"

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

# A token goes to curl in a header file, never on a command line.
auth=()
gh_curl() { curl -fsSL --retry 3 --retry-delay 2 "${auth[@]}" "$@"; }

api="https://api.github.com/repos/${repo}/releases/tags/${tag}"
if ! gh_curl -H 'Accept: application/vnd.github+json' -o "${work}/release.json" "${api}" 2> "${work}/err"; then
    token="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
    [[ -n "${token}" ]] || die "cannot read release '${tag}' of ${repo}: $(< "${work}/err")"
    printf 'Authorization: Bearer %s\n' "${token}" > "${work}/auth"
    auth=(-H "@${work}/auth")
    gh_curl -H 'Accept: application/vnd.github+json' -o "${work}/release.json" "${api}" \
        || die "cannot read release '${tag}' of ${repo}, with or without a token"
fi

# name, size, digest, download URL and API URL of each wheel, tab-separated.
# An asset name is the file name written below, so only plain ones pass.
python3 -I - "${work}/release.json" > "${work}/assets" << 'PY'
import json
import re
import sys

release = json.load(open(sys.argv[1], encoding="utf-8"))
print(
    f"release {release['tag_name']}: published {release.get('published_at')}, "
    f"source {release.get('target_commitish')}",
    file=sys.stderr,
)
for asset in release.get("assets", []):
    name = asset["name"]
    if not name.endswith(".whl"):
        continue
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", name):
        sys.exit(f"refusing asset with an unexpected name: {name!r}")
    print(name, asset["size"], asset.get("digest") or "-", asset["browser_download_url"], asset["url"], sep="\t")
PY

# verify FILE SIZE DIGEST: the size, and the sha256 when the release has one.
verify() {
    [[ "$(stat -c %s "$1")" == "$2" ]] || return 1
    [[ "$3" == sha256:* ]] || return 0
    [[ "$(sha256sum "$1" | cut -d' ' -f1)" == "${3#sha256:}" ]]
}

mkdir -p "${dest}"
declare -A listed=()
while IFS=$'\t' read -r name size digest url api_url; do
    listed["${name}"]=1
    file="${dest}/${name}"
    if [[ -f "${file}" ]] && verify "${file}" "${size}" "${digest}"; then
        echo "  kept      ${name}"
        continue
    fi
    echo "  download  ${name} ($((size / 1048576)) MiB)"
    if [[ ${#auth[@]} -gt 0 ]]; then
        gh_curl -H 'Accept: application/octet-stream' -o "${file}.part" "${api_url}"
    else
        gh_curl -o "${file}.part" "${url}"
    fi
    if ! verify "${file}.part" "${size}" "${digest}"; then
        rm -f "${file}.part"
        die "${name} does not match the size or sha256 the release publishes"
    fi
    mv -f "${file}.part" "${file}"
done < "${work}/assets"
[[ ${#listed[@]} -gt 0 ]] || die "release '${tag}' of ${repo} has no .whl assets"

shopt -s nullglob
for file in "${dest}"/*.whl; do
    if [[ -z "${listed[${file##*/}]+x}" ]]; then
        echo "  removed   ${file##*/} (not in release '${tag}')"
        rm -f "${file}"
    fi
done
echo "${#listed[@]} wheel(s) in ${dest}"
