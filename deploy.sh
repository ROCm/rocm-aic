#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Sync the Grafana "ROCm-AIC cluster summary" dashboard with the repo and
# cluster. Optional file deploy.local.env (gitignored) may set GRAFANA_URL,
# GRAFANA_ADMIN_USER, GRAFANA_ADMIN_PASSWORD.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -f "${REPO_ROOT}/deploy.local.env" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/deploy.local.env"
fi

DASHBOARD_UID=${DASHBOARD_UID:-rocm-aic-cluster}
DASHBOARD_JSON=${DASHBOARD_JSON:-"${REPO_ROOT}/ansible/roles/monitoring_stack/files/grafana-rocm-aic-cluster.json"}
GRAFANA_ADMIN_USER=${GRAFANA_ADMIN_USER:-admin}
GRAFANA_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:-rocm-aic}

usage() {
  cat <<'EOF'
Usage: ./deploy.sh [command]

Commands (default: all):
  pull   GET dashboard JSON from Grafana API -> update local file
  bump   Increment the "version" field in the local dashboard JSON
  apply  Run Ansible monitoring play (prometheus rules + provisioned JSON)
  push   POST local dashboard JSON to Grafana API (overwrite)
  all    pull, then bump, then apply, then push

Environment:
  GRAFANA_URL            Base URL (default: derived from ansible/inventory/hosts.yml)
  GRAFANA_ADMIN_USER     (default: admin)
  GRAFANA_ADMIN_PASSWORD (default: rocm-aic for this lab cluster)
  DASHBOARD_UID          (default: rocm-aic-cluster)
  DASHBOARD_JSON         Path to dashboard file in repo
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "deploy.sh: missing required command: $1" >&2
    exit 1
  fi
}

grafana_url() {
  if [[ -n "${GRAFANA_URL:-}" ]]; then
    echo "${GRAFANA_URL%/}"
    return
  fi
  local inv="${REPO_ROOT}/ansible/inventory/hosts.yml"
  if [[ ! -f "${inv}" ]]; then
    echo "deploy.sh: set GRAFANA_URL or add ansible/inventory/hosts.yml" >&2
    exit 1
  fi
  python3 - "${inv}" <<'PY'
import re
import sys
path = sys.argv[1]
text = open(path, encoding="utf-8").read()
m = re.search(r"monitoring_server:\s*\n\s+hosts:\s*\n\s+(\S+):", text)
if not m:
    print("http://127.0.0.1:3000", end="")
    sys.exit(0)
host = m.group(1)
m2 = re.search(rf"{re.escape(host)}:\s*\n\s+ansible_host:\s*(\S+)", text)
ip = m2.group(1) if m2 else host
print(f"http://{ip}:3000", end="")
PY
}

cmd_pull() {
  require_cmd curl
  require_cmd jq
  local base url tmp
  base=$(grafana_url)
  url="${base}/api/dashboards/uid/${DASHBOARD_UID}"
  tmp=$(mktemp)
  trap 'rm -f "${tmp}"' EXIT
  echo "deploy.sh: pulling ${url}" >&2
  curl -sS -f -u "${GRAFANA_ADMIN_USER}:${GRAFANA_ADMIN_PASSWORD}" "${url}" >"${tmp}"
  jq '.dashboard | del(.id)' "${tmp}" | jq '.' >"${DASHBOARD_JSON}"
  rm -f "${tmp}"
  trap - EXIT
  echo "deploy.sh: wrote ${DASHBOARD_JSON}" >&2
}

cmd_bump() {
  require_cmd jq
  local tmp
  tmp=$(mktemp)
  jq '.version = ((.version // 0) | tonumber) + 1' "${DASHBOARD_JSON}" >"${tmp}"
  mv -f "${tmp}" "${DASHBOARD_JSON}"
  echo "deploy.sh: bumped version in ${DASHBOARD_JSON} to $(jq -r '.version' "${DASHBOARD_JSON}")" >&2
}

cmd_apply() {
  require_cmd ansible-playbook
  echo "deploy.sh: running Ansible monitoring play" >&2
  (
    cd "${REPO_ROOT}/ansible"
    ansible-playbook -i inventory/hosts.yml playbooks/monitoring.yml --tags monitoring
  )
}

cmd_push() {
  require_cmd curl
  require_cmd jq
  local base tmp
  base=$(grafana_url)
  tmp=$(mktemp)
  trap 'rm -f "${tmp}"' EXIT
  jq -n --slurpfile d "${DASHBOARD_JSON}" \
    '{dashboard: $d[0], overwrite: true, message: "deploy.sh push"}' >"${tmp}"
  echo "deploy.sh: pushing to ${base}/api/dashboards/db" >&2
  curl -sS -f -u "${GRAFANA_ADMIN_USER}:${GRAFANA_ADMIN_PASSWORD}" \
    -H 'Content-Type: application/json' \
    -X POST \
    --data-binary "@${tmp}" \
    "${base}/api/dashboards/db" | jq .
  rm -f "${tmp}"
  trap - EXIT
}

cmd_all() {
  cmd_pull
  cmd_bump
  cmd_apply
  cmd_push
}

main() {
  local sub=${1:-all}
  case "${sub}" in
    -h | --help | help)
      usage
      ;;
    pull) cmd_pull ;;
    bump) cmd_bump ;;
    apply) cmd_apply ;;
    push) cmd_push ;;
    all) cmd_all ;;
    *)
      echo "deploy.sh: unknown command: ${sub}" >&2
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"
