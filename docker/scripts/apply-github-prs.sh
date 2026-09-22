#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Cherry-pick GitHub pull request heads listed in a manifest onto a local repo.
# Supports stacked PRs: prerequisite commits not already present on HEAD are
# cherry-picked before dependent commits in ancestor order.
set -euo pipefail

if (( BASH_VERSINFO[0] < 4 )); then
	echo "ERROR: apply-github-prs.sh requires bash 4 or newer" >&2
	exit 1
fi

REPO_DIR="${1:?repo dir is required}"
MANIFEST_PATH="${2:?manifest path is required}"
BASE_REF_NAME="${3:?base ref name is required}"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
	echo "ERROR: ${REPO_DIR} is not a git repository" >&2
	exit 1
fi

if [[ ! -f "${MANIFEST_PATH}" ]]; then
	exit 0
fi

is_supported_github_remote_url() {
	local remote_url="${1}"
	[[ "${remote_url}" =~ ^https://github\.com/[^/[:space:]]+/[^/[:space:]]+(\.git)?$ ]] \
		|| [[ "${remote_url}" =~ ^git@github\.com:[^/[:space:]]+/[^/[:space:]]+(\.git)?$ ]] \
		|| [[ "${remote_url}" =~ ^ssh://git@github\.com/[^/[:space:]]+/[^/[:space:]]+(\.git)?$ ]]
}

git -C "${REPO_DIR}" config user.name "AIC Image Build"
git -C "${REPO_DIR}" config user.email "rocm-aic@noreply.invalid"

base_commit="$(git -C "${REPO_DIR}" rev-parse "${BASE_REF_NAME}^{commit}")"

while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
	line="$(sed -e 's/#.*//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' <<<"${raw_line}")"
	if [[ -z "${line}" ]]; then
		continue
	fi

	read -r -a parts <<<"${line}"
	if [[ "${#parts[@]}" -eq 1 ]]; then
		remote_spec=origin
		pr_number="${parts[0]}"
	elif [[ "${#parts[@]}" -eq 2 ]]; then
		remote_spec="${parts[0]}"
		pr_number="${parts[1]}"
	else
		echo "ERROR: invalid PR manifest entry: ${raw_line}" >&2
		echo "Expected '<pr-number>', '<remote-name> <pr-number>', or '<remote-name>=<remote-url> <pr-number>'." >&2
		exit 1
	fi

	if [[ ! "${pr_number}" =~ ^[0-9]+$ ]]; then
		echo "ERROR: invalid PR number in entry: ${raw_line}" >&2
		exit 1
	fi

	remote_name="${remote_spec%%=*}"
	remote_url=""
	if [[ "${remote_spec#*=}" != "${remote_spec}" ]]; then
		remote_url="${remote_spec#*=}"
		if ! is_supported_github_remote_url "${remote_url}"; then
			echo "ERROR: remote '${remote_name}' must use a supported GitHub URL (https://github.com/..., git@github.com:..., or ssh://git@github.com/...)." >&2
			exit 1
		fi
	fi

	existing_remote_url=""
	if existing_remote_url="$(git -C "${REPO_DIR}" remote get-url "${remote_name}" 2>/dev/null)"; then
		if [[ -n "${remote_url}" && "${existing_remote_url}" != "${remote_url}" ]]; then
			echo "ERROR: remote '${remote_name}' already points to '${existing_remote_url}', not '${remote_url}'." >&2
			exit 1
		fi
	else
		if [[ -z "${remote_url}" ]]; then
			echo "ERROR: remote '${remote_name}' is not configured; use '<remote-name>=<remote-url> <pr-number>'." >&2
			exit 1
		fi
		git -C "${REPO_DIR}" remote add "${remote_name}" "${remote_url}"
	fi

	pr_ref="refs/remotes/${remote_name}/pr/${pr_number}"
	if ! git -C "${REPO_DIR}" fetch "${remote_name}" \
		"+refs/pull/${pr_number}/head:${pr_ref}"; then
		echo "ERROR: PR ${remote_name}#${pr_number} could not be fetched via refs/pull/${pr_number}/head; use a remote that exposes GitHub-style PR refs." >&2
		exit 1
	fi

	merge_base="$(git -C "${REPO_DIR}" merge-base "${base_commit}" "${pr_ref}")"
	if [[ "${merge_base}" != "${base_commit}" ]]; then
		echo "ERROR: PR ${remote_name}#${pr_number} does not descend from ${BASE_REF_NAME}; rebase it or convert it to a patch." >&2
		exit 1
	fi

	declare -A needed_commits=()
	while read -r cherry_status cherry_commit; do
		if [[ "${cherry_status}" == "+" ]]; then
			needed_commits["${cherry_commit}"]=1
		fi
	done < <(git -C "${REPO_DIR}" cherry HEAD "${pr_ref}")

	mapfile -t candidate_commits < <(
		git -C "${REPO_DIR}" rev-list --reverse "${base_commit}..${pr_ref}"
	)
	pr_commits=()
	for pr_commit in "${candidate_commits[@]}"; do
		if [[ -n "${needed_commits["${pr_commit}"]:-}" ]]; then
			pr_commits+=("${pr_commit}")
		fi
	done

	if [[ "${#pr_commits[@]}" -eq 0 ]]; then
		echo "Skipped PR ${remote_name}#${pr_number}: already included"
		continue
	fi

	if git -C "${REPO_DIR}" cherry-pick -x "${pr_commits[@]}"; then
		echo "Applied PR ${remote_name}#${pr_number}"
	else
		git -C "${REPO_DIR}" cherry-pick --abort >/dev/null 2>&1 || true
		echo "ERROR: PR ${remote_name}#${pr_number} does not cherry-pick cleanly" >&2
		exit 1
	fi
done < "${MANIFEST_PATH}"
