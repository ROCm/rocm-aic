#!/usr/bin/env bash
# -E inherit ERR trap in functions and subshells,
# -e exit the script upon commands failing,
# -u to error if the script ever references a variable that is unset,
# -o pipefail to error if any stage of a pipeline fails.
set -Eeuo pipefail

# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# errexit exits without saying where; report it.  Only the line number and
# function call stack are printed, never the command text, which can carry
# secret values.
_on_err() {
    # set -E carries this trap into $(...), where errexit is off.
    [[ $- == *e* ]] || return 0
    # || : so a failed write cannot replace the exit status.
    printf '%s: command failed (exit %s, pipeline status %s) at line %s%s\n' \
        "${0##*/}" "$1" "$2" "$3" "${4:+ in ${4// / <- }}" >&2 || :
}
trap '_on_err "$?" "${PIPESTATUS[*]}" "$LINENO" "${FUNCNAME[*]-}"' ERR

# Replace a rolling GitHub release (e.g. `nightly`) without ever leaving it with
# a partial asset set, keeping its stable releases/download/<tag>/ URLs.
#
#   stage   <tag> <target-sha> <title> <notes-file> <wheel-dir>
#       Upload every <wheel-dir>/*.whl to a new draft release
#       (<tag>-staging-<run>-<attempt>, no git tag yet), check every asset
#       landed with the local size, and print the draft's tag on stdout.  The
#       live <tag> release is untouched, so a failed upload costs nothing.
#   promote <tag> <staging-tag> <target-sha>
#       Delete the live <tag> release and its git tag, then publish the staged
#       draft as <tag> at <target-sha>.  The only window with no <tag> release is
#       between those two API calls.
#
# Stage every release a run publishes before promoting any of them, so a failed
# build or upload in one leaves all of them on the previous run's assets.
# Needs GH_TOKEN (contents: write) and GITHUB_REPOSITORY.

log() { printf '%s\n' "$*" >&2; }

repo="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY must be set}"

stage() {
    local tag="$1" sha="$2" title="$3" notes="$4" dir="$5"
    local staging="${tag}-staging-${GITHUB_RUN_ID:-$(date +%s)}-${GITHUB_RUN_ATTEMPT:-1}"
    local stale old remote expected w

    [[ -r "${notes}" ]] || { log "publish: cannot read notes file ${notes}"; return 1; }
    shopt -s nullglob
    local wheels=("${dir}"/*.whl)
    [[ ${#wheels[@]} -gt 0 ]] || { log "publish: no wheels in ${dir}"; return 1; }

    # Drafts an earlier failed run of this tag left behind.  They hold no git
    # tag, so deleting them cannot affect the live release.
    stale="$(gh release list --repo "${repo}" --limit 200 --json tagName,isDraft \
        -q ".[] | select(.isDraft and (.tagName | startswith(\"${tag}-staging-\"))) | .tagName")"
    while IFS= read -r old; do
        [[ -n "${old}" ]] || continue
        log "publish: deleting stale staging draft ${old}"
        gh release delete "${old}" --repo "${repo}" --yes
    done <<<"${stale}"

    log "publish: staging ${#wheels[@]} wheel(s) as draft ${staging}"
    gh release create "${staging}" "${wheels[@]}" \
        --repo "${repo}" \
        --draft \
        --prerelease \
        --title "${title}" \
        --notes-file "${notes}" \
        --target "${sha}" >&2

    # Every local wheel, by name and size, must be an uploaded asset, and
    # nothing else may be.
    expected="$(for w in "${wheels[@]}"; do
        printf '%s %s\n' "${w##*/}" "$(stat -c %s "${w}")"
    done | sort)"
    remote="$(gh release view "${staging}" --repo "${repo}" --json assets \
        -q '.assets[] | select(.state == "uploaded") | "\(.name) \(.size)"' | sort)"
    if [[ "${remote}" != "${expected}" ]]; then
        log "publish: draft ${staging} assets do not match the local wheels (< local, > draft):"
        diff <(printf '%s\n' "${expected}") <(printf '%s\n' "${remote}") >&2 || :
        return 1
    fi
    log "publish: draft ${staging} verified"
    printf '%s\n' "${staging}"
}

promote() {
    local tag="$1" staging="$2" sha="$3" out published

    if out="$(gh release view "${tag}" --repo "${repo}" --json tagName -q .tagName 2>&1)"; then
        log "publish: deleting live release ${tag}"
        gh release delete "${tag}" --repo "${repo}" --yes --cleanup-tag
    elif [[ "${out}" == *"release not found"* ]]; then
        log "publish: no live ${tag} release"
    else
        log "publish: cannot look up release ${tag}: ${out}"
        return 1
    fi

    # A tag with no release (an interrupted earlier promote) would pin the new
    # release to the old commit, since publishing reuses an existing tag.
    if out="$(gh api "repos/${repo}/git/ref/tags/${tag}" -q .object.sha 2>&1)"; then
        log "publish: deleting leftover tag ${tag} (${out})"
        gh api -X DELETE "repos/${repo}/git/refs/tags/${tag}" >/dev/null
    elif [[ "${out}" != *"HTTP 404"* ]]; then
        log "publish: cannot look up tag ${tag}: ${out}"
        return 1
    fi

    if ! gh release edit "${staging}" --repo "${repo}" \
        --tag "${tag}" --target "${sha}" --draft=false --prerelease >&2; then
        log "publish: ERROR: ${tag} is now absent; the verified wheels are still in draft ${staging}."
        log "publish: publish it by hand: gh release edit ${staging} --repo ${repo} --tag ${tag} --target ${sha} --draft=false --prerelease"
        return 1
    fi

    published="$(gh api "repos/${repo}/git/ref/tags/${tag}" -q .object.sha)"
    if [[ "${published}" != "${sha}" ]]; then
        log "publish: ERROR: tag ${tag} points at ${published}, expected ${sha}"
        return 1
    fi
    log "publish: ${tag} published at ${sha}"
}

case "${1:-}" in
    stage)
        [[ $# -eq 6 ]] || { log "usage: $0 stage <tag> <target-sha> <title> <notes-file> <wheel-dir>"; exit 2; }
        stage "$2" "$3" "$4" "$5" "$6"
        ;;
    promote)
        [[ $# -eq 4 ]] || { log "usage: $0 promote <tag> <staging-tag> <target-sha>"; exit 2; }
        promote "$2" "$3" "$4"
        ;;
    *)
        log "usage: $0 stage <tag> <target-sha> <title> <notes-file> <wheel-dir>"
        log "       $0 promote <tag> <staging-tag> <target-sha>"
        exit 2
        ;;
esac
